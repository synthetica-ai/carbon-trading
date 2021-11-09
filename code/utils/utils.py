import numpy as np
import tensorflow as tf
import pandas as pd


def cii_expected(dwt, year=2023):
    """
    a function calculating the annual required cii of a vessel given a year
    """
    a = 4745
    c = 0.622
    z = {2023: 5, 2024: 7, 2025: 9, 2026: 8}
    cii_ref = a * dwt ** (-c)
    cii_required = ((100 - z[year]) / 100) * cii_ref
    return cii_required


def cii_attained(dwt, distance_sailed, co2_emissions):
    """
    a function calculating the annual attained cii of a vessel
    """
    cii_attained = (co2_emissions * 1_000_000) / (dwt * distance_sailed)
    return cii_attained


def cii_rating(attained_cii, required_cii):
    """
    a function calculating the cii rating of a vessel
    """
    d1, d2, d3, d4 = 0.86, 0.94, 1.06, 1.18
    rating = attained_cii / required_cii
    if rating <= d1:
        return "Rating A"
    elif rating <= d2:
        return "Rating B"
    elif rating <= d3:
        return "Rating C"
    elif rating <= d4:
        return "Rating D"
    else:
        return "Rating E"


def can_reach(vessel_location, vessel_set_of_speeds, start_port, start_time, free, previous_end_port):
    """
    function checking if a vessel can reach a start port on time
    """
    # if the vessel is free
    # the vessel has to reach the new start port on time
    # u = Dx / Dt

    # # if the vessel is not free
    # the vessel has to reach the previous end port on time # use set of speeds

    # u = Dx / Dt
    # and then
    # the vessel has to reach the new start port on time

    pass


def can_serve(vessels_df, contract_df, ports_df):
    """
    function checking if a vessel meets the conditions to serve a contract
    """
    # # Check if vessel has the capacity to transport the cargo
    # vessel_capacity >= cargo_volume

    #     # If it does then check if it is free or already serving a cargo
    #     vessel_free == Yes:

    #         # Check if the vessel can reach the start port at start_time
    #         can_reach()

    pass


def create_tensor_dm(dm_df):
    """
    `create_tensor_dm` produces a tf tensor out of the distance matrix dataframe
    Args :
    * dm_df : A dataframe containing the distance matrix data
    """
    dist_cols = dm_df.columns.to_list()
    del dist_cols[0]
    dm_array = dm_df.loc[:, dist_cols].to_numpy()
    dm_tensor = tf.convert_to_tensor(dm_array)
    return dm_tensor


def find_distance(port_1_number, port_2_number, dm):
    """
    `find_distance` returns the distance between two ports
    Args:
    * port_1_number : number of port 1
    * port_2_number : number of port 2
    * dm : distance matrix array or tensor
    """
    dist_m = dm
    idx_1 = port_1_number - 1
    idx_2 = port_2_number - 1
    distance = dist_m[idx_1, idx_2]
    return distance


def func_ballast(con_tensor, ships_tensor, dm_tensor):
    dm = dm_tensor.numpy()
    sp_idx = con_tensor[:, 0] - 1
    cols_idx = sp_idx.numpy().astype(int)

    # current_port idxs
    cp_idx = ships_tensor[:, 4] - 1
    rows_idx = cp_idx.numpy().astype(int)

    # get ballast data from distance matrix
    bd = dm[np.ix_(rows_idx, cols_idx)]
    # convert to tf tensor
    bd = tf.convert_to_tensor(bd)
    # extend the ballast data
    bd_ext = tf.concat([bd] * 3, axis=1)
    # get rid of the first column of bd
    # this step is needed for proper alignment of the mask later
    bd_ext = bd_ext[:, 1:]
    # cast the extended ballast data to float
    bd_ext = tf.cast(bd_ext, dtype=float)
    # number of fleet features
    num_ship_feats = 11
    num_contracts = 4
    ones_col_dim = num_ship_feats - num_contracts
    # creating an array of ones and zeros for the mask
    ones_and_zeros = tf.concat(
        [tf.ones([num_contracts, ones_col_dim]), tf.zeros([num_contracts, num_contracts]),], axis=1,
    )
    # casting the ones and zeros to boolean to create the boolean mask
    mask = tf.cast(ones_and_zeros, dtype="bool")

    # where mask == true take fleet data else take ext_ballast data
    fleet_with_ballast = tf.where(mask, ships_tensor, bd_ext)
    return fleet_with_ballast

