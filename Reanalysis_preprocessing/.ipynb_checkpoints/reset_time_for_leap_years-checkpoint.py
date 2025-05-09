import xarray as xr
import numpy as np
import os

import argparse

parser = argparse.ArgumentParser(description="Process year and month.")
parser.add_argument("--year", type=str, required=True, help="Year to process.")

args = parser.parse_args()

year = args.year

def is_leap_year(year):
    return (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0))

files=[f"sst.{year}.nc",
       f"ice.{year}.nc",
       f"lwdn.{year}.nc",
       f"swdn.{year}.nc",
       f"precip_rate.{year}.nc",
       f"t_10.{year}.nc",
       f"q_10.{year}.nc",
       f"u_10.{year}.nc",
       f"v_10.{year}.nc"]



for file in files:
    
    input_file = file
    output_file = file + ".fixed"

    if not os.path.exists(input_file):
        print(f"File {input_file} does not exist. Skipping...")
        continue

    print(input_file)
    
    ds = xr.open_dataset(input_file)
    num_time_steps = len(ds["time"])

    new_time = np.arange(num_time_steps)

    print("assigning new coords")
    ds = ds.assign_coords(time=("time", new_time))

    print("writing to file")
    ds.to_netcdf(output_file)

    print(f"Time variable fixed and saved to {output_file}")





