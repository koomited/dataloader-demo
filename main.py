import json
import time
from typing import Optional

import dask
import torch
import typer
import xbatcher
import xarray as xr
from arraylake import Client, config
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from torch import multiprocessing
from typing_extensions import Annotated
from torch.utils.data._utils.collate import default_collate


from dask.cache import Cache

from model import VariationalAutoEncoder
from tqdm import tqdm
from torch import nn, optim


# comment these the next two lines out to disable Dask's cache
cache = Cache(1e10)  # 10gb cache
cache.register()

# Training constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4 # Karpathy constant

loss_fn = nn.MSELoss()


def training(model, batch_generator,
             optimizer,
             reconst_loss_fn = loss_fn,
             num_epochs: int = NUM_EPOCHS,
             device = DEVICE
             ):
    for epoch in range(num_epochs):
        
        loop = tqdm(enumerate(batch_generator))
        loop = tqdm(batch_generator, desc=f"Epoch [{epoch+1}/{num_epochs}]")
        
        
        for i, (x, dates) in enumerate(loop):
            # print(x.shape)
            input = x[:, :2, :, :, :]
            input= input.to(device)
            
            # input = input.squeeze(0).squeeze(1)
            input = input.squeeze(0)
            input = torch.nan_to_num(input, nan=0.0)
            
            
            target = x[:, 2:, :, :, :]
            target_for_now = target.squeeze(0)
            
            target_for_now = target_for_now[:2,:,:]
            
            target_for_now = torch.nan_to_num(target_for_now, nan=0.0)
            
            
            # print(f"Input shape:{input.shape}")
            
            # predictions = model(input)[0]
            x_recon, mu, sigma = model(input)
            
            reconst_loss = reconst_loss_fn(x_recon, target_for_now)
            kl_div = -torch.sum(1+torch.log(sigma.pow(2))-mu.pow(2)-sigma.pow(2))

            loss = reconst_loss + kl_div
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss = loss.item())
            
      

def print_json(obj):
    print(json.dumps(obj))


class XBatcherPyTorchDataset(TorchDataset):
    def __init__(self, batch_generator: xbatcher.BatchGenerator):
        self.bgen = batch_generator

    def __len__(self):
        return len(self.bgen)

    def __getitem__(self, idx):
        t0 = time.time()
        print_json(
            {
                "event": "get-batch start",
                "time": t0,
                "idx": idx,
                "pid": multiprocessing.current_process().pid,
            }
        )
        # load before stacking
        batch = self.bgen[idx].load()
        dates = batch["time"].values  

        # Use to_stacked_array to stack without broadcasting,
        stacked = batch.to_stacked_array(
            new_dim="batch", sample_dims=("time", "longitude", "latitude")
        ).transpose("time", "batch", ...)
        x = torch.tensor(stacked.data)
        # x = dict(
        #     times = times,
        #     data = x
        #     )
        t1 = time.time()
        print_json(
            {
                "event": "get-batch end",
                "time": t1,
                "idx": idx,
                "pid": multiprocessing.current_process().pid,
                "duration": t1 - t0,
            }
        )
        return (x, dates)


def setup(source="gcs", patch_size: int = 4, input_steps: int = 10, local_path: str = None):
    if source == "gcs":
        ds = xr.open_dataset(
            "gs://weatherbench2/datasets/era5/1959-2022-6h-128x64_equiangular_with_poles_conservative.zarr",
            engine="zarr",
            chunks={},
        )
    elif source == "arraylake":
        config.set({"s3.endpoint_url": "https://storage.googleapis.com", "s3.anon": True})
        ds = (
            Client()
            .get_repo("earthmover-public/weatherbench2")
            .to_xarray(
                group="datasets/era5/1959-2022-6h-128x64_equiangular_with_poles_conservative",
                chunks={},
            )
        )
    elif source == "local_nc":
        ds = xr.open_dataset(
            local_path,
            engine="h5netcdf",
            chunks={},
        )
    else:
        raise ValueError(f"Unknown source {source}")

    DEFAULT_VARS = [
        "precipitation_amount",
        # "10m_wind_speed",
        # "2m_temperature",
        # "specific_humidity",
    ]

    ds = ds[DEFAULT_VARS]
    
    num_longitudes = len(ds.longitude) - len(ds.longitude) % patch_size
    num_latitudes = len(ds.latitude) - len(ds.latitude) % patch_size
    patch = dict(
        latitude=num_latitudes,
        longitude=num_longitudes,
        time=input_steps,
    )
    overlap = dict(latitude=32, longitude=32, time=input_steps // 3 * 2)

    bgen = xbatcher.BatchGenerator(
        ds,
        input_dims=patch,
        input_overlap=overlap,
        preload_batch=False,
    )
    

    dataset = XBatcherPyTorchDataset(bgen)

    return dataset


def custom_collate_fn(batch):
    """
    Custom collate that keeps datetime64 arrays as-is.
    """
    xs, dates = zip(*batch)
    # Default collate for tensors
    xs = default_collate(xs)
    # Keep dates as list or numpy arrays
    return xs, dates[0]


def main(
    source: Annotated[str, typer.Option()] = "arraylake",
    num_epochs: Annotated[int, typer.Option(min=0, max=1000)] = 2,
    num_batches: Annotated[int, typer.Option(min=0, max=1000)] = 3,
    batch_size: Annotated[int, typer.Option(min=0, max=1000)] = 16,
    shuffle: Annotated[Optional[bool], typer.Option()] = None,
    num_workers: Annotated[Optional[int], typer.Option(min=0, max=64)] = None,
    prefetch_factor: Annotated[Optional[int], typer.Option(min=0, max=64)] = None,
    persistent_workers: Annotated[Optional[bool], typer.Option()] = None,
    pin_memory: Annotated[Optional[bool], typer.Option()] = None,
    train_step_time: Annotated[Optional[float], typer.Option()] = 0.1,
    dask_threads: Annotated[Optional[int], typer.Option()] = None,
    local_path: Annotated[Optional[str], typer.Option()] = None,
    
):
    _locals = {k: v for k, v in locals().items() if not k.startswith("_")}
    data_params = {
        "batch_size": batch_size,
    }
    if shuffle is not None:
        data_params["shuffle"] = shuffle
    if num_workers is not None:
        data_params["num_workers"] = num_workers
        data_params["multiprocessing_context"] = "forkserver"
    if prefetch_factor is not None:
        data_params["prefetch_factor"] = prefetch_factor
    if persistent_workers is not None:
        data_params["persistent_workers"] = persistent_workers
    if pin_memory is not None:
        data_params["pin_memory"] = pin_memory
    if dask_threads is None or dask_threads <= 1:
        dask.config.set(scheduler="single-threaded")
    else:
        dask.config.set(scheduler="threads", num_workers=dask_threads)

    run_start_time = time.time()
    print_json(
        {
            "event": "run start",
            "time": run_start_time,
            "data_params": str(data_params),
            "locals": _locals,
        }
    )

    t0 = time.time()
    print_json({"event": "setup start", "time": t0})
    dataset= setup(source=source, local_path=local_path)
    # training_generator = DataLoader(dataset, **data_params)
    training_generator = DataLoader(dataset, collate_fn=custom_collate_fn, **data_params)

    _ = next(iter(training_generator))  # wait until dataloader is ready
    t1 = time.time()
    print_json({"event": "setup end", "time": t1, "duration": t1 - t0})

    model = VariationalAutoEncoder()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    

    training(model = model, 
             optimizer=optimizer,
            batch_generator = training_generator,
            reconst_loss_fn = loss_fn,
            num_epochs = num_epochs,
            device = DEVICE
        )
    # for epoch in range(num_epochs):
    #     e0 = time.time()
    #     print_json({"event": "epoch start", "epoch": epoch, "time": e0})

    #     # for i, sample in enumerate(training_generator):
        
    #     for i, (x, dates) in enumerate(training_generator):
    #         # print(x.shape)
    #         input = x[:, :2, :, :, :]
    #         input_dates = dates[:2]
    #         target_dates = dates[2:]
    #         target = x[:, 2:, :, :, :]
            
    #         # input = input.squeeze(0).squeeze(1)
    #         input = input.squeeze(0)
    #         target = target.squeeze(0).squeeze(1)
            
    #         target_for_now = target[:2,:,:]
    #         print(f"Input shape:{input.shape}")
            
    #         predictions = model(input)[0]

    #         print(f"Prediction shape:{predictions.shape}")
    #         # print(f"Tragte shape:{target_for_now.shape}")
            
    #         # print(sample)
    #         # print(sample.shape)
    #         tt0 = time.time()
    #         print_json({"event": "training start", "batch": i, "time": tt0})
    #         time.sleep(train_step_time)  # simulate model training
    #         tt1 = time.time()
    #         print_json({"event": "training end", "batch": i, "time": tt1, "duration": tt1 - tt0})
    #         if i == num_batches - 1:
    #             break

    #     e1 = time.time()
    #     print_json({"event": "epoch end", "epoch": epoch, "time": e1, "duration": e1 - e0})

    # run_finish_time = time.time()
    # print_json(
    #     {"event": "run end", "time": run_finish_time, "duration": run_finish_time - run_start_time}
    # )


if __name__ == "__main__":
    typer.run(main)
