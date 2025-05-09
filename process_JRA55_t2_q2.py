import xarray as xr
import numpy as np
import xesmf as xe
from shift_t2_q2_neutral_winds import convert_winds_real2neutral

# Year range
START_YEAR = 1960
END_YEAR = 1970

# Paths and filenames
PATH_MASK = '/glade/campaign/cgd/oce/people/rita/JRA55/landsea_mask/'
PATH_SST = '/glade/campaign/cgd/oce/people/rita/COBESST/'
PATH_ICE = '/glade/campaign/cgd/oce/people/rita/COBESST2/'
MASK_FILE = 'JRA55_grid_file.nc'
FILENAME_SST = 'COBESST_189101_202306_Rita_20241101.nc'
FILENAME_ICE = 'icec.mon.mean.nc'

# Load the mask once
print("Loading land-sea mask...")
ds_mask = 1 - xr.open_dataset(PATH_MASK + MASK_FILE).tmask.values  # Swap 0 and 1

# Initialize regridder (created for the first year and reused)
regridder = None

for YEAR in range(START_YEAR, END_YEAR + 1):
    print(f"\nProcessing year: {YEAR}")

    # Paths for the current year
    PATH_JRA = f"/glade/campaign/cgd/oce/people/rita/JRA55/fcst_surf/{YEAR}/"
    forcing_name_start = "fcst_surf."
    forcing_name_end = f".reg_tl319.{YEAR}.nc"
    filenames_forcing = ["011_t_2", "051_q_2", "033_u_10", "034_v_10", "002_slp"]

    # Load forcing datasets dynamically
    forcing_data = xr.Dataset()
    
    print("Reading forcing datasets...")
    for var, var_name in zip(filenames_forcing, ["t_2", "q_2", "u_10", "v_10", "slp"]):
        filepath = PATH_JRA + forcing_name_start + var + forcing_name_end
        ds = xr.open_dataset(filepath)
        forcing_data[var_name] = ds[var_name]
        forcing_data[var_name]["time"] = forcing_data[var_name].indexes["time"]

    
    # Load SST data
    print("Reading SST and sea ice datasets...")
    ds_sst = xr.open_dataset(PATH_SST + FILENAME_SST).sel(time=slice(f"{YEAR}-01-01", f"{YEAR+1}-01-01"))

    if regridder is None:  # Create regridder only once
        regridder = xe.Regridder(ds_sst, forcing_data["t_2"], method="bilinear")
        print('regridder created')

    # Regrid SST and interpolate to hourly time points
    sst = regridder(ds_sst["SST_cpl"]).interp(time=forcing_data["t_2"]["time"])

    # Load and regrid sea ice
    ds_ice = xr.open_dataset(PATH_ICE + FILENAME_ICE).sel(time=slice(f"{YEAR}-01-01", f"{YEAR+1}-01-01"))
    ice = regridder(ds_ice["icec"]).interp(time=forcing_data["t_2"]["time"])

    # Convert to NumPy arrays
    print("Converting data to NumPy arrays...")
    t2, q2, u10, v10, slp = (forcing_data[key].values for key in ["t_2", "q_2", "u_10", "v_10", "slp"])
    sst, ice = sst.values, ice.values
    mask = ds_mask

    # Initialize output arrays
    t_10 = np.empty_like(t2)
    q_10 = np.empty_like(q2)

    # Process each timestep
    print("Processing timesteps...")
    for itime in range(t2.shape[0]):
        if itime % 50 == 0:
            print(f"Processing timestep {itime}/{t2.shape[0]}")
        wnd_tmp = np.sqrt(u10[itime] ** 2 + v10[itime] ** 2)
        t2_tmp, q2_tmp, slp_tmp, sst_tmp, ice_tmp = t2[itime], q2[itime], slp[itime], sst[itime], ice[itime]
        t_10[itime], q_10[itime] = convert_winds_real2neutral(wnd_tmp, t2_tmp, q2_tmp, slp_tmp, sst_tmp, ice_tmp, mask)

    # Create and save t_10 dataset
    print("Saving t_10 dataset...")
    t_10_ds = xr.Dataset(
        {"t_10": (["time", "latitude", "longitude"], t_10, {"units": "K", "description": "Temperature at 10m"})},
        coords={
            "time": forcing_data["t_2"].coords["time"],
            "latitude": forcing_data["t_2"].coords["latitude"],
            "longitude": forcing_data["t_2"].coords["longitude"],
        },
        attrs={"description": "Neutral wind-adjusted field for T10"}
    )
    t_10_ds.to_netcdf(f"{PATH_JRA}fcst_surf.t_10.reg_tl319.{YEAR}.nc")

    # Create and save q_10 dataset
    print("Saving q_10 dataset...")
    q_10_ds = xr.Dataset(
        {"q_10": (["time", "latitude", "longitude"], q_10, {"units": "kg/kg", "description": "Specific humidity at 10m"})},
        coords={
            "time": forcing_data["t_2"].coords["time"],
            "latitude": forcing_data["t_2"].coords["latitude"],
            "longitude": forcing_data["t_2"].coords["longitude"],
        },
        attrs={"description": "Neutral wind-adjusted field for Q10"}
    )
    q_10_ds.to_netcdf(f"{PATH_JRA}fcst_surf.q_10.reg_tl319.{YEAR}.nc")

    print(f"Year {YEAR} processing complete.\n")

print("All years processed successfully.")
