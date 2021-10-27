def create_contracts(seed):
    """
    A function that creates 4 random cargo contracts at each env step.\
    There are 5 cargo contract types handy,supra,ultra,pana,kamsar for each port.\
    The function :
    * returns a 1d array of 50 elements
    * all array elements are integers from 0 to 4
    * the sum of the array must be 4
    """
    import numpy as np
    number_of_contracts = 4
    np.random.seed(seed)
    x = np.random.multinomial(number_of_contracts, [1/5]*5, None)
    z = np.zeros(shape=45, dtype=np.int32)
    c = np.concatenate((x, z))
    np.random.shuffle(c)
    return c


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


def find_distance():
 # get distance between start and end ports arrays
    start_ports_index = df['start_port'] - 1
    print(start_ports_index)
    end_ports_index = df['end_port']
    print(end_ports_index)
    dist_df = dm.iloc[start_ports_index, end_ports_index]
    # the distance
    df['distance'] = pd.Series(np.diag(dist_df)).reindex()
    pass
