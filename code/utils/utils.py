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


def find_cii_attained(ship_number, speed, distance):
    """
    `find_cii_attained` calculates the attained cii of a ship

    Inputs

    ship_number: the number of the ship [1,2,3,4]
    speed : the speed of the ship
    distance : the distance covered
    """

    speed_factors_dict_per_ship_type = {
        1: (50_000, 0.0008_5034, 0.1564_6258),
        2: (60_000, 0.0075_7758, 0.0316_6888),
        3: (70_000, 0.0073_5917, 0.0817_2108),
        4: (80_000, 0.0073_5917, 0.0817_2108),
    }

    # print(f"to distance sailed einai {distance}")
    # dwt
    dwt = speed_factors_dict_per_ship_type[ship_number][0]
    # print(f"To dwt einai {dwt}")

    # the factor of speed^3
    c3_speed_factor = speed_factors_dict_per_ship_type[ship_number][1]
    # print(f"O c3 speed factor einai {c3_speed_factor}")
    # the factor of speed^2
    c2_speed_factor = speed_factors_dict_per_ship_type[ship_number][2]
    # print(f"O c2 speed factor einai {c2_speed_factor}")

    # co2 emissions from speed formula
    co2_emissions = (c3_speed_factor * (speed ** 3)) + (c2_speed_factor * (speed ** 2))
    # print(f"Ta co2 emissions einai {co2_emissions}")

    numerator = co2_emissions * 1_000_000
    # print(f"O ari8mhths einai {numerator}")
    denomenator = dwt * distance
    # print(f"O paronomasths einai {denomenator}")
    cii_attained = numerator / denomenator
    return cii_attained


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


def func_ballast(con_tensor, ships_tensor, dm_tensor):

    dm = dm_tensor.numpy()
    sp_idistance = con_tensor[:, 0] - 1
    cols_idistance = sp_idistance.numpy().astype(int)

    # current_port idistances
    cp_idistance = ships_tensor[:, 4] - 1
    rows_idistance = cp_idistance.numpy().astype(int)

    # get ballast data from distance matrix
    bd = dm[np.ix_(rows_idistance, cols_idistance)]
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


def map_action(selected_action):
    """
    `map_action` maps the selected action `selected_action` to contract number and speed

    Inputs : The selected action `selected_action`
    Outputs: The selected contract `selected_contract` and the selected speed `selected speed`
    """
    possible_contracts = np.array([0, 1, 2, 3])
    possible_speeds = np.array([10, 12, 14])
    # print(f"The selected action is {selected_action}")
    if selected_action not in np.arange(0, 13):
        # print(f"Your selected action {selected_action} is out of the possible bounds")
        # print("")
        selected_contract = "Out of bounds"
        selected_speed = "Out of bounds"
    elif selected_action == 12:
        selected_contract = None
        selected_speed = 0
        # print(f"This means you did not select any contract")
        # print(f"The selected speed is {selected_speed} knots")
        # print("")

    else:
        selected_contract = possible_contracts[selected_action // 3]
        selected_speed = possible_speeds[selected_action % 3]
        # print(f"The selected contract is contract {selected_contract+1}")
        # print(f"The selected speed is {selected_speed} knots")
        # print("")
    return selected_contract, selected_speed


def find_duration(u, distance):
    """
    `find_duration` finds the duration of a trip with distance `distance` conducted with speed `u` in `days`
    """
    dt_hours = tf.math.round((distance / u))
    dt_days = tf.math.round((dt_hours / 24))
    return dt_days, dt_hours


def prepare_ships_log(ships_log):
    """
    `prepare_ships_log` changes the ships_log and returns the available ships for the day
    """

    # check ships_log
    # afairese apo ka8e entry tou ships_log (days_of_unavailability) -1 efoson phgame sthn epomenh mera
    # an prokupsoun arnhtika values kanta 0
    # dhmiourghse mia lista available_ships pou 8a periexei ta ships pou einai available (days_of_unavailability = 0)

    available_ships = []
    for k in ships_log:
        if ships_log[k] <= 0:
            ships_log[k] = 0
            available_ships.append(k)

        # afairese apo ola ta ship logs mia mera unavailability
        ships_log[k] -= 1
    return ships_log, available_ships


def generate_state_at_new_day(env, available_ships_list):
    """
    `generate_state_at_new_day` creates the state at a new day

    It needs to generate:

    * 4 new contracts
    * contracts_mask me asous pantou
    * ships tensor me bash ta available ships
    * ships mask me bash ta available ships
    """

    ships_tensor = env.ships_tensor
    # print(f"o ships tensor einai {ships_tensor}")
    # pairnw ta idxs twn available ships
    # available_ships_idx = [x - 1 for x in available_ships_list]

    available_ships_idx = [x for x in available_ships_list]
    inplace_array = np.zeros(4)
    # bazw sta idxs twn available ships asso
    inplace_array[available_ships_idx] = 1

    # ftiaxnw enan indices_array (4,11) me bash to shape tou ships_tensor
    indices_array = np.zeros(shape=(ships_tensor.shape))
    # epeidh thelw na allaksw ston ship_tensor to ship_availability (feature 6) kanw auth th sthlh 1
    indices_array[:, 6] = 1

    # pairnw to index tou ship_availability feature se morfh tensora
    # tha parw ena tensora me shape (4,2) kai morfh [[0,6],[1,6],[2,6],[3,6]]
    indices = tf.cast(tf.where(tf.equal(indices_array, 1)), tf.int32)

    # ftiaxnw enan scattered array
    # sta indices pou ypodeiknyei o indices bazw tis times tou inplace array
    scatter = tf.scatter_nd(indices, inplace_array, shape=ships_tensor.shape)

    # o inverse_mask einai o logikos antistrofos tou indices_array
    inverse_mask = tf.cast(tf.math.logical_not(indices_array), tf.float32)

    # o input_array_zero exei 0 stis 8eseis pou thelw na allaksw kai se oles tis alles theseis exei thn timh tou ships_tensor
    input_array_zero_out = tf.multiply(inverse_mask, ships_tensor)

    # o ananewmenos ship tensor me tis times pou exei o scatter tensor ekei pou htan ta mhdenika tou input_array_zero_out
    ships_tensor_updated = tf.add(input_array_zero_out, tf.cast(scatter, tf.float32))

    # bazw ton ananewmeno ships_tensor san attribute sto environment
    ships_tensor = tf.cast(ships_tensor_updated, tf.float32)

    contracts_df, contracts_tensor, = env.create_contracts_tensor()

    # Add the ballast distances to the ships tensor
    ships_tensor = func_ballast(
        con_tensor=contracts_tensor, ships_tensor=ships_tensor, dm_tensor=env.dm_tensor,
    )

    # contracts_mask = contracts_tensor[:, 7]

    # ships_mask = ships_tensor[:, 6]

    state_dict = {"contracts_state": contracts_tensor, "ships_state": ships_tensor}

    return state_dict


# region

# def cii_rating(attained_cii, required_cii):
#     """
#     a function calculating the cii rating of a vessel
#     """
#     d1, d2, d3, d4 = 0.86, 0.94, 1.06, 1.18
#     rating = attained_cii / required_cii
#     if rating <= d1:
#         return "Rating A"
#     elif rating <= d2:
#         return "Rating B"
#     elif rating <= d3:
#         return "Rating C"
#     elif rating <= d4:
#         return "Rating D"
#     else:
#         return "Rating E"

# def find_distance(port_1_number, port_2_number, dm):
# """
# `find_distance` returns the distance between two ports
# Args:
# * port_1_number : number of port 1
# * port_2_number : number of port 2
# * dm : distance matrix array or tensor
# """
# dist_m = dm
# idistance_1 = port_1_number - 1
# idistance_2 = port_2_number - 1
# distance = dist_m[idistance_1, idistance_2]
# return distance


# def can_reach(vessel_location, vessel_set_of_speeds, start_port, start_time, free, previous_end_port):
#     """
#     function checking if a vessel can reach a start port on time
#     """
#     # if the vessel is free
#     # the vessel has to reach the new start port on time
#     # u = distance / Dt

#     # # if the vessel is not free
#     # the vessel has to reach the previous end port on time # use set of speeds

#     # u = distance / Dt
#     # and then
#     # the vessel has to reach the new start port on time

#     pass


# def can_serve(vessels_df, contract_df, ports_df):
#     """
#     function checking if a vessel meets the conditions to serve a contract
#     """
#     # # Check if vessel has the capacity to transport the cargo
#     # vessel_capacity >= cargo_volume

#     #     # If it does then check if it is free or already serving a cargo
#     #     vessel_free == Yes:

#     #         # Check if the vessel can reach the start port at start_time
#     #         can_reach()

#     pass

# endregion

