import gym
from gym import spaces
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.ops import Tensor
from utils.utils import cii_expected, func_ballast


class CarbonEnv(gym.Env):
    """
    Description :

    A custom openai gym environment for the carbon emission problem.

    """

    def __init__(
        self,
        data_dict={
            "ships_path": "data/fleet_small.csv",
            "ports_path": "data/ports_10.csv",
            "dm_path": "data/distance_matrix.csv",
        },
    ):
        super().__init__()

        self.data_dict = data_dict

        # get fleet info in df
        self.ships = pd.read_csv(self.data_dict["ships_path"])

        # get port info in df
        ports = pd.read_csv(self.data_dict["ports_path"])
        self.ports = ports.loc[:, ["number", "name", "country"]]

        # get distance matrix info
        self.dm_df = pd.read_csv(self.data_dict["dm_path"])

        # get distance matrix as tensor
        self.dm_tensor = self.create_dm_tensor()

        self.NUM_SHIPS = len(self.ships)
        self.NUM_PORTS = len(self.ports)
        self.NUM_DAILY_CONTRACTS = 4
        self.SET_OF_SPEEDS = [10, 12, 14]
        self.NUM_SPEEDS = len(self.SET_OF_SPEEDS)
        self.NUM_CONTRACT_FEATURES = 10
        self.NUM_SHIP_FEATURES = 11
        self.batch_size = 32
        self.observation_space = {
            "contracts_state": tf.zeros(shape=(self.NUM_DAILY_CONTRACTS, self.NUM_CONTRACT_FEATURES)),
            "ships_state": tf.zeros(shape=(self.NUM_SHIPS, self.NUM_SHIP_FEATURES)),
            "contracts_mask": tf.zeros(shape=(self.NUM_DAILY_CONTRACTS, 1)),
            "ships_mask": tf.zeros(shape=(self.NUM_SHIPS, 1)),
        }
        self.observation_space_concatenated = tf.concat(
            (
                self.observation_space["contracts_state"],
                self.observation_space["ships_state"],
                self.observation_space["contracts_mask"],
                self.observation_space["ships_mask"],
            ),
            axis=1,
        )
        self.observation_space_dim = self.observation_space_concatenated.shape
        self.observation_space_dim = self.observation_space_dim.as_list()
        self.action_space = {"actions": tf.zeros(shape=((self.NUM_DAILY_CONTRACTS * self.NUM_SPEEDS) + 1, 1))}
        self.action_space_dim = self.action_space["actions"].shape
        self.action_space_dim = self.action_space_dim.as_list()
        self.embedding_size = 128
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

        # prepei na kanw to step gia ka8e ship

        state, reward, done = self.state, 1, False

        # state = self.contracts_tensor

        # reward =

        return state, reward, done

    def reset(self):
        """
        `reset` sets the environment to its initial state

        Returns:
        * initial_state : the initial state / observation of the environment.

        """

        self.day = 0
        self.info = {}
        self.done = False

        # Set the fleet to its initial state
        self.ships = pd.read_csv(self.data_dict["ships_path"])

        # Calculate fleet's required cii
        self.ships["cii_threshold"] = self.ships["dwt"].map(cii_expected)

        # set fleet at random ports
        self.ships["current_port"] = np.random.randint(1, self.NUM_PORTS + 1, self.NUM_DAILY_CONTRACTS)

        # create a fleet tensor from the fleet df
        self.ships_tensor = self.create_ships_tensor()

        # Create the contracts for the first day of the year
        self.contracts_df, self.contracts_tensor = self.create_contracts_tensor()

        # Add the ballast distances to the fleet tensor
        self.ships_tensor = func_ballast(
            con_tensor=self.contracts_tensor, ships_tensor=self.ships_tensor, dm_tensor=self.dm_tensor,
        )

        # An entity showing which daily contracts were taken (1) and which were not (0)
        self.contracts_mask = tf.ones(shape=(self.NUM_DAILY_CONTRACTS, 1))

        # An entity showing for how many days each ship is to be reserved
        # reserve_duration = (balast_distance of that contract + contract_distance) / picked_speed
        self.ship_log = np.zeros(shape=(self.NUM_SHIPS, 1))

        # should stay 1 for as long as this ship is reserved
        self.ships_mask = tf.ones(shape=(self.NUM_SHIPS, 1))

        self.state = {
            "contracts_state": self.contracts_tensor,
            "ships_state": self.ships_tensor,
            "contracts_mask": self.contracts_mask,
            "ships_mask": self.ships_mask,
        }

        return self.state

    def create_contracts(self, seed=None):
        """
        `create_contracts` creats cargo contracts for a specific day of the year
        """
        con_df = pd.DataFrame(
            columns=[
                "start_port_number",
                "end_port_number",
                "contract_type",
                "start_day",
                "end_day",
                "cargo_size",
                "contract_duration",
                "contract_availability",
                "contract_distance",
                "value",
            ]
        )

        ship_types = np.array(["supramax", "ultramax", "panamax", "kamsarmax"])

        con_df["start_port_number"] = np.random.randint(1, self.NUM_PORTS + 1, size=self.NUM_DAILY_CONTRACTS)
        con_df["contract_type"] = np.random.choice(ship_types, size=self.NUM_DAILY_CONTRACTS)
        con_df["end_port_number"] = np.random.randint(1, self.NUM_PORTS + 1, size=self.NUM_DAILY_CONTRACTS)

        same_ports = con_df["start_port_number"] == con_df["end_port_number"]
        # check that start and end ports are different
        while sum(same_ports) != 0:
            con_df["end_port_number"] = np.where(
                same_ports,
                np.random.randint(low=1, high=self.NUM_PORTS + 1, size=same_ports.shape),
                con_df["end_port_number"],
            )
            same_ports = con_df["start_port_number"] == con_df["end_port_number"]

        con_df["start_day"] = self.day

        # get distance between start and end ports arrays
        start_port_numbers_index = con_df["start_port_number"] - 1
        end_port_numbers_index = con_df["end_port_number"]

        dist_df = self.dm_df.iloc[start_port_numbers_index, end_port_numbers_index]

        # the distance
        con_df["contract_distance"] = pd.Series(np.diag(dist_df)).reindex()

        # Create cargo size based on ship_type
        type_conditions = [
            con_df["contract_type"] == "supramax",
            con_df["contract_type"] == "ultramax",
            con_df["contract_type"] == "panamax",
            con_df["contract_type"] == "kamsarmax",
        ]

        cargo_size_choices = [
            np.random.randint(40_000, 50_000, type_conditions[0].shape),
            np.random.randint(50_000, 60_000, type_conditions[1].shape),
            np.random.randint(60_000, 70_000, type_conditions[2].shape),
            np.random.randint(70_000, 80_000, type_conditions[3].shape),
        ]

        con_df["cargo_size"] = np.select(type_conditions, cargo_size_choices)

        ship_type_to_ship_code_choices = [
            np.ones(shape=type_conditions[0].shape),
            2 * np.ones(shape=type_conditions[1].shape),
            3 * np.ones(shape=type_conditions[2].shape),
            4 * np.ones(shape=type_conditions[3].shape),
        ]

        con_df["contract_type"] = np.select(type_conditions, ship_type_to_ship_code_choices)

        # calculate duration

        # pick random speed from possible set of speeds
        u_picked = np.random.choice([10, 12, 14])

        # pick distance between ports from df
        dx = con_df["contract_distance"]
        # find duration of trip between ports with picked speed in hours
        dt_hours = (dx / u_picked).round()
        # find duration of trip between ports in days
        dt_days = (dt_hours / 24).round()

        # get upper triangle entries of distance matrix
        x = self.dm_df.iloc[:, 1:].to_numpy(dtype=np.int32)
        mask_upper = np.triu_indices_from(x, k=1)
        triu = x[mask_upper]
        # average voyage distance between ports in the distance matrix
        avg_dx = np.round(triu.mean())
        # average voyage duration between ports with picked speed in hours
        avg_dt_hours = np.round(avg_dx / u_picked)
        # # average voyage duration between ports with picked speed in days
        avg_dt_days = np.round(avg_dt_hours / 24)

        # total duration

        con_df["contract_duration"] = dt_days + avg_dt_days

        # end_day ends at 23:59
        con_df["end_day"] = con_df["start_day"] + con_df["contract_duration"] - 1

        # add contract value
        con_df["value"] = round(
            con_df["cargo_size"] * (con_df["contract_distance"] / (con_df["contract_duration"] * 1_000_000))
        )

        # set contract availability to 1 for each contract
        con_df["contract_availability"] = np.ones(shape=(self.NUM_DAILY_CONTRACTS))

        return con_df

    def create_contracts_tensor(self):
        """
        `create_contracts_tensor` creates a tensor out of the contracts dataframe
        """
        empty = pd.DataFrame(
            columns=[
                "start_port_number",
                "end_port_number",
                "contract_type",
                "start_day",
                "end_day",
                "cargo_size",
                "contract_duration",
                "contract_availability",
                "contract_distance",
                "value",
            ]
        )
        contracts_df = empty.copy()
        x = self.create_contracts()
        contracts_df = contracts_df.append(x, ignore_index=True)

        # convert everything to float for tensorflow compatibility
        contracts_df = contracts_df.astype(np.float32)

        # create the input tensor
        contracts_tensor = tf.convert_to_tensor(contracts_df)

        # add a batch size dimension
        # contracts_tensor = tf.expand_dims(contracts_tensor, axis=0)

        return contracts_df, contracts_tensor

    def create_ships_tensor(self):
        """
        `create_ships_tensor` creates a tensor out of the fleets dataframe
        """
        # keeping only these features from the fleet df
        cols_to_keep = [
            "ship_number",
            "dwt",
            "cii_threshold",
            "cii_attained",
            "current_port",
            "current_speed",
            "ship_availability",
            "ballast_1",
            "ballast_2",
            "ballast_3",
            "ballast_4",
        ]

        self.ships = self.ships[cols_to_keep]

        df = self.ships

        # converting to float for tensorflow compatibility
        df = df.astype(np.float32)

        # create the tensor
        tensor = tf.convert_to_tensor(df)

        # add a batch size dimension
        # tensor = tf.expand_dims(tensor, axis=0)

        return tensor

    def create_dm_tensor(self):
        """
        `create_dm_tensor` produces a tf tensor out of the distance matrix dataframe
        Args :
        * dm_df : A dataframe containing the distance matrix data
        """
        dist_cols = self.dm_df.columns.to_list()
        del dist_cols[0]
        dm_array = self.dm_df.loc[:, dist_cols].to_numpy()
        dm_tensor = tf.convert_to_tensor(dm_array)
        return dm_tensor

    def find_distance(self, port_1_number, port_2_number):
        """
        `find_distance` returns the distance between two ports
        Args:
        * port_1_number : number of port 1
        * port_2_number : number of port 2
        """
        dist_m = self.dm
        idx_1 = port_1_number - 1
        idx_2 = port_2_number - 1
        distance = dist_m[idx_1, idx_2]
        return distance

