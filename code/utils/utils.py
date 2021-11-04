import numpy as np
import tensorflow as tf
import pandas as pd


def create_contracts_old(dm, ports, day=1, seed=None):
    """
    A function for creating cargo contracts for a specific day of the year
    """
    con_df = pd.DataFrame(columns=['start_port_number', 'end_port_number', 'contract_type',
                          'start_day', 'end_day', 'cargo_size', 'contract_duration', 'port_distance', 'value'])
    ports = ports.number.to_numpy()
    ship_types = np.array(['supramax', 'ultramax', 'panamax', 'kamsarmax'])
    con_df['start_port_number'] = np.repeat(ports, 4)
    con_df['contract_type'] = np.tile(ship_types, 10)
    con_df['end_port_number'] = np.random.randint(low=1, high=11, size=(40,))
    same_ports = con_df['start_port_number'] == con_df['end_port_number']
    while sum(same_ports) != 0:
        con_df['end_port_number'] = np.where(same_ports, np.random.randint(
            low=1, high=11, size=same_ports.shape), con_df['end_port_number'])
        same_ports = con_df['start_port_number'] == con_df['end_port_number']
    con_df['start_day'] = day

    # get distance between start and end ports arrays
    start_port_numbers_index = con_df['start_port_number'] - 1
    end_port_numbers_index = con_df['end_port_number']

    dist_df = dm.iloc[start_port_numbers_index,
                      end_port_numbers_index]
    # the distance
    con_df['port_distance'] = pd.Series(np.diag(dist_df)).reindex()

    # Create cargo size based on ship_type
    type_conditions = [con_df['contract_type'] == 'supramax',
                       con_df['contract_type'] == 'ultramax',
                       con_df['contract_type'] == 'panamax',
                       con_df['contract_type'] == 'kamsarmax']

    cargo_size_choices = [np.random.randint(40_000, 50_000, type_conditions[0].shape),
                          np.random.randint(
                              50_000, 60_000, type_conditions[1].shape),
                          np.random.randint(
                              60_000, 70_000, type_conditions[2].shape),
                          np.random.randint(70_000, 80_000, type_conditions[3].shape)]

    con_df['cargo_size'] = np.select(type_conditions, cargo_size_choices)

    ship_type_to_ship_code_choices = [np.ones(shape=type_conditions[0].shape),
                                      2*np.ones(shape=type_conditions[1].shape),
                                      3*np.ones(shape=type_conditions[2].shape),
                                      4*np.ones(shape=type_conditions[3].shape)]

    con_df['contract_type'] = np.select(
        type_conditions, ship_type_to_ship_code_choices)

    # calculate duration

    # pick random speed from possible set of speeds
    u_picked = np.random.choice([10, 12, 14])

    # pick distance between ports from df
    dx = con_df['port_distance']
    # find duration of trip between ports with picked speed in hours
    dt_hours = (dx / u_picked).round()
    # find duration of trip between ports in days
    dt_days = (dt_hours / 24).round()

    # get upper triangle entries of distance matrix
    x = dm.iloc[:, 1:].to_numpy(dtype=np.int32)
    mask_upper = np.triu_indices_from(x, k=1)
    triu = x[mask_upper]
    # average voyage distance between ports in the distance matrix
    avg_dx = np.round(triu.mean())
    # average voyage duration between ports with picked speed in hours
    avg_dt_hours = np.round(avg_dx/u_picked)
    # # average voyage duration between ports with picked speed in days
    avg_dt_days = np.round(avg_dt_hours / 24)

    # total duration

    con_df['contract_duration'] = dt_days + avg_dt_days

    # end_day ends at 23:59
    con_df['end_day'] = con_df['start_day'] + con_df['contract_duration'] - 1

    # add contract value
    con_df['value'] = round(
        con_df['cargo_size'] * (con_df['port_distance'] / (con_df['contract_duration'] * 1_000_000)))
    return con_df


def cii_expected(dwt, year=2023):
    """
    a function calculating the annual required cii of a vessel given a year
    """
    a = 4745
    c = 0.622
    z = {2023: 5, 2024: 7, 2025: 9, 2026: 8}
    cii_ref = a * dwt ** (-c)
    cii_required = ((100 - z[year])/100) * cii_ref
    return cii_required


def cii_attained(dwt, distance_sailed, co2_emissions):
    """
    a function calculating the annual attained cii of a vessel
    """
    cii_attained = (co2_emissions * 1_000_000)/(dwt*distance_sailed)
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
    idx_1 = port_1_number-1
    idx_2 = port_2_number-1
    distance = dist_m[idx_1, idx_2]
    return distance


# # create random contracts old
# def create_contrs(self=None, df=contracts_df, dm = distance_matrix , step=1, seed=None):
#         """
#         A method that creates 4 random cargo contracts at each env step. \
#         There are 5 cargo contract types handy,supra,ultra,pana,kamsar for each port.\
#         The function :
#         * returns a 1d array of 4 elements
#         * all array elements are integers from 0 to 4
#         * the sum of the array must be 4
#         """
#         day = step
#         # creating 4 contracts for the specific day
#         contr_count = 4
#         np.random.seed(seed)
#         # x array of 5 elements that sum to one
#         x = np.random.multinomial(contr_count, [1/5]*5, None)
#         # y array of 45 zeros
#         z = np.zeros(shape=45, dtype=np.int32)
#         # contracts of size 50 concatenation of x,y
#         contracts = np.concatenate((x,z))
#         np.random.shuffle(contracts)

#         # getting port locations where contracts have spawned
#         locations = np.nonzero(contracts)[0]
#         # getting the start ports of the contract
#         ports = np.floor_divide(locations,5)
#         # ports start at port 1
#         ports = ports + 1
#         # getting the contract count for a specific location
#         contract_count_per_loc = contracts[locations]
#         # array of fives to help with modulo
#         fives = np.ones((len(locations)))*5
#         # find ship type at port with modulo
#         x = np.mod(locations,fives)
#         # creating the ship_conditions for each shiptype
#         ship_conditions = [ x == 0 , x == 1, x == 2, x == 3, x == 4 ]
#         ship_choices = ['handymax','supramax','ultramax','panamax','kamsarmax']
#         # getting the shiptype and converting it to str
#         ship_types = np.select(ship_conditions,ship_choices)
#         ship_types = ship_types.astype(str)
#         # populating the df
#         df['contract_type'] = np.repeat(ship_types,contract_count_per_loc)
#         df['start_port_number'] = np.repeat(ports,contract_count_per_loc)
#         df['end_port_number'] = np.random.randint(low=1, high=11, size=(4,))
#         # Check for contracts with same start and end ports
#         same_ports = df['start_port_number'] == df['end_port_number']
#         # repeat until no start and end ports of the same contract are the same
#         while sum(same_ports) != 0 :
#           df['end_port_number'] = np.where(same_ports, np.random.randint(low=1, high=11, size=same_ports.shape), df['end_port_number'])
#           same_ports = df['start_port_number'] == df['end_port_number']

#         # get distance between start and end ports arrays
#         start_port_numbers_index = df['start_port_number'] - 1

#         end_port_numbers_index = df['end_port_number']

#         dist_df = dm.iloc[start_port_numbers_index,end_port_numbers_index]
#         # the distance
#         df['port_distance'] = pd.Series(np.diag(dist_df)).reindex()


#         # Create cargo size based on ship_type
#         type_conditions = [ df['contract_type'] == 'handymax',
#                             df['contract_type'] == 'supramax',
#                             df['contract_type'] == 'ultramax',
#                             df['contract_type'] == 'panamax',
#                             df['contract_type'] == 'kamsarmax']

#         cargo_size_choices = [np.random.randint(30_000,40_000,type_conditions[0].shape),
#                               np.random.randint(40_000,50_000,type_conditions[1].shape),
#                               np.random.randint(50_000,60_000,type_conditions[2].shape),
#                               np.random.randint(60_000,70_000,type_conditions[3].shape),
#                               np.random.randint(70_000,80_000,type_conditions[4].shape)]

#         df['cargo_size'] = np.select(type_conditions,cargo_size_choices)

#         # add contract value
#         df['value'] = np.round(df['cargo_size'] * 1.5)

#         # add start day
#         df['start_day'] = day


#         # calculate duration

#         # pick random speed from possible set of speeds
#         u_picked = np.random.choice([10,12,14])

#         # pick distance between ports from df
#         dx = df['port_distance']
#         # find duration of trip between ports with picked speed in hours
#         dt_hours = ( dx / u_picked).round()
#         # find duration of trip between ports in days
#         dt_days = (dt_hours / 24).round()

#         # get upper triangle entries of distance matrix
#         x = dm.iloc[:,1:].to_numpy(dtype=np.int32)
#         mask_upper = np.triu_indices_from(x,k=1)
#         triu = x[mask_upper]
#         # average voyage distance between ports in the distance matrix
#         avg_dx = np.round(triu.mean())
#         # average voyage duration between ports with picked speed in hours
#         avg_dt_hours = np.round(avg_dx/u_picked)
#         # # average voyage duration between ports with picked speed in days
#         avg_dt_days = np.round(avg_dt_hours / 24)

#         # total duration
#         df['contract_duration'] = dt_days + avg_dt_days


#         # end_day ends at 23:59
#         df['end_day'] = df['start_day'] + df['contract_duration'] - 1

#         return df


# create tensor contracts old

# empty = pd.DataFrame(columns=['start_port_number','end_port_number','contract_type','start_day','end_day','cargo_size','contract_duration','port_distance','value'])
# contracts_df = empty.copy()
# for i in range(1,365+1):
#     x = create_contracts(day=i)
#     contracts_df = contracts_df.append(x, ignore_index=True)

# # convert everything to float for tensorflow compatibility
# contracts_df = contracts_df.astype(np.float32)

# # create the input tensor
# contracts_tensor = tf.convert_to_tensor(contracts_df)

# # add a batch size dimension
# contracts_tensor = tf.expand_dims(contracts_tensor,axis=0)

# # contracts tensor is the input tensor
