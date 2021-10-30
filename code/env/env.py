import gym
from gym import spaces
from gym.spaces.discrete import Discrete
import pandas as pd
import numpy as np
import tensorflow as tf
from utils.utils import cii_expected


class CarbonEnv(gym.Env):
    """
    Description :

    A custom openai gym environment for the carbon emission problem.

    """

    def __init__(self, data_dict):
        super().__init__()

        self.data = data_dict
        # get fleet info
        self.fleet = pd.read_csv(self.data['fleet_path'])
        # get port info
        self.ports = pd.read_csv(self.data['ports_path'])
        # get distance matrix
        self.dm = pd.read_csv(self.data['dm_path'])

        NUM_SHIPS = len(self.fleet)
        NUM_PORTS = len(self.ports)
        NUM_DAILY_CONTRACTS = NUM_SHIPS * NUM_PORTS
        SET_OF_SPEEDS = [10, 12, 14]
        NUM_SPEEDS = len(SET_OF_SPEEDS)

        # the observation space changes daily based on the step==1 day
        observation_space = spaces.Dict({
            "contracts": spaces.Discrete(NUM_DAILY_CONTRACTS,),
            "ships": spaces.Discrete(NUM_SHIPS,)
        })

        # action_space = spaces.Dict({
        #     # "choose_ship": spaces.Discrete(NUM_SHIPS+1),
        #     # we loop on every ship and take actions on each of the ships
        #     # using +1 to account for the case of not choosing a contract
        #     # for each ship we must choose:
        #     # * which contract to take among the available which are at most NUM_CONTRACTS+1
        #     # * which speed to use for the trip
        #     "choose_contract": spaces.Discrete(NUM_YEARLY_CONTRACTS+1),
        #     "choose_speed": spaces.Discrete(NUM_SPEEDS+1)
        # })

        # The action space should be described in a daily manner as well
        action_space = spaces.Dict({
            # we loop over the ships for the contracts of the day specified by the step
            # the actions we can take for each ship are:
            "choose_contract": spaces.Discrete(NUM_DAILY_CONTRACTS),
            "choose_speed": spaces.Discrete(NUM_SPEEDS)

        })

        self.reset()

    def step(self, action):
        """
        `step` takes a step into the environment

        Returns:
        * obs: The observation produced by the action of the agent
        * reward: The reward produced by the action of the agent
        * done: A flag signaling if the game ended
        * info : A dict useful for debugging
        """

        pass

    def reset(self):
        """
        `reset` sets the environment to its initial state

        Returns:
        * initial_state : the initial state / observation of the environment.

        """

        self.info = {}
        self.done = False

        # Set the fleet to its initial state
        self.fleet = pd.read_csv(self.data['fleet_path'])

        # Calculate fleet's required cii
        self.fleet['cii_threshold'] = self.fleet['dwt'].map(cii_expected)

        # set fleet at random ports
        self.fleet['current_port'] = np.random.randint(
            1, 11, self.fleet.shape[0])

        # create a fleet tensor from the fleet df
        self.fleet_tensor = self.create_tensor_fleet()

        # Create the contract tensor for the whole year
        self.con_tensor = self.create_tensor_contracts()

        # These contracts must be all passed to the mlp encoder

        # Moreover as part of the initial observation I should get:

        # 1. The 40 first tensor contracts
        # 2. The fleet tensor

        # Getting the 40 first tensor contract
        self.idx_min, self.idx_max = 0, 40
        self.con_first_40 = self.con_tensor[:, self.idx_min:self.idx_max, :]

        # the ships are also part of the initial state / observation

        initial_state = {"contracts": self.con_first_40,
                         "ships": self.fleet_tensor}

        return initial_state

    def create_contracts(self, day=1, seed=None):
        """
        `create_contracts` creats cargo contracts for a specific day of the year
        """
        con_df = pd.DataFrame(columns=['start_port_number', 'end_port_number', 'contract_type',
                              'start_day', 'end_day', 'cargo_size', 'contract_duration', 'port_distance', 'value'])
        ports = self.ports.number.to_numpy()
        ship_types = np.array(['supramax', 'ultramax', 'panamax', 'kamsarmax'])
        con_df['start_port_number'] = np.repeat(ports, 4)
        con_df['contract_type'] = np.tile(ship_types, 10)
        num_contracts = len(ship_types) * len(ports)
        con_df['end_port_number'] = np.random.randint(
            low=1, high=11, size=(num_contracts,))
        same_ports = con_df['start_port_number'] == con_df['end_port_number']
        while sum(same_ports) != 0:
            con_df['end_port_number'] = np.where(same_ports, np.random.randint(low=1, high=11, size=same_ports.shape),
                                                 con_df['end_port_number'])
            same_ports = con_df['start_port_number'] == con_df['end_port_number']

        # setting the start_day to the current day
        con_df['start_day'] = day

        # get distance between start and end ports arrays
        start_port_numbers_index = con_df['start_port_number'] - 1
        end_port_numbers_index = con_df['end_port_number']
        dist_df = self.dm.iloc[start_port_numbers_index,
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
        x = self.dm.iloc[:, 1:].to_numpy(dtype=np.int32)
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
        con_df['end_day'] = con_df['start_day'] + \
            con_df['contract_duration'] - 1

        # add contract value
        con_df['value'] = round(
            con_df['cargo_size'] * (con_df['port_distance'] / (con_df['contract_duration'] * 1_000_000)))

        return con_df

    def create_tensor_contracts(self):
        """
        `create_tensor_contracts` creates a tensor out of the contracts dataframe
        """
        empty = pd.DataFrame(columns=['start_port_number', 'end_port_number', 'contract_type',
                             'start_day', 'end_day', 'cargo_size', 'contract_duration', 'port_distance', 'value'])
        contracts_df = empty.copy()
        for i in range(1, 365+1):
            x = self.create_contracts(day=i)
            contracts_df = contracts_df.append(x, ignore_index=True)

        # convert everything to float for tensorflow compatibility
        contracts_df = contracts_df.astype(np.float32)

        # create the input tensor
        contracts_tensor = tf.convert_to_tensor(contracts_df)

        # add a batch size dimension
        contracts_tensor = tf.expand_dims(contracts_tensor, axis=0)

        return contracts_tensor

    def create_tensor_fleet(self):
        """
        `create_tensor_fleet` creates a tensor out of the fleets dataframe
        """
        # keeping only these features from the fleet df
        cols_to_keep = ['ship_number', 'dwt', 'cii_threshold', 'cii',
                        'current_port', 'current_speed', 'ship_availability']
        df = self.fleet[cols_to_keep]

        # converting to float for tensorflow compatibility
        df = df.astype(np.float32)

        # create the tensor
        tensor = tf.convert_to_tensor(df)

        # add a batch size dimension
        tensor = tf.expand_dims(tensor, axis=0)

        return tensor
