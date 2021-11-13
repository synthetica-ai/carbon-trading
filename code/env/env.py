from re import S
from typing_extensions import ParamSpecArgs, ParamSpecKwargs
import gym
from gym import spaces
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.op_callbacks import should_invoke_op_callbacks
from tensorflow.python.framework.ops import Tensor
from utils.utils import (
    cii_expected,
    find_cii_attained,
    func_ballast,
    map_action,
    find_duration,
)


class CarbonEnv(gym.Env):
    """
    Description :

    A custom openai gym environment for the carbon emission problem.

    """

    def __init__(
        self, data_dict={"ships_path": "data/fleet_small.csv", "ports_path": "data/ports_10.csv", "dm_path": "data/distance_matrix.csv",},
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
            "contracts_state": tf.zeros(shape=(self.NUM_DAILY_CONTRACTS, self.NUM_CONTRACT_FEATURES,)),
            "ships_state": tf.zeros(shape=(self.NUM_SHIPS, self.NUM_SHIP_FEATURES)),
            "contracts_mask": tf.zeros(shape=(self.NUM_DAILY_CONTRACTS, 1)),
            "ships_mask": tf.zeros(shape=(self.NUM_SHIPS, 1)),
        }
        self.observation_space_concatenated = tf.concat(
            (self.observation_space["contracts_state"], self.observation_space["ships_state"], self.observation_space["contracts_mask"], self.observation_space["ships_mask"]), axis=1,
        )

        self.observation_space_dim = self.observation_space_concatenated.shape.as_list()
        # self.observation_space_dim = self.observation_space_dim.as_list()
        self.action_space = {"actions": tf.zeros(shape=((self.NUM_DAILY_CONTRACTS * self.NUM_SPEEDS) + 1, 1,))}
        # self.action_space = {"actions": tf.zeros(shape=(self.NUM_SPEEDS + 1, 1))}
        self.action_space_dim = self.action_space["actions"].shape.as_list()
        # self.action_space_dim = self.action_space_dim.as_list()
        self.embedding_size = 128
        self.reset()

    def step(self, ship_number, action):
        """
        `step` takes a step into the environment

        Returns:
        * obs: The observation produced by the action of the agent
        * reward: The reward produced by the action of the agent
        * done: A flag signaling if the game ended
        * info : A dict useful for debugging
        """
        ### Prosoxh!!!!
        # to ship_number einai [1,2,3,4] den ksekinaei apo to 0!
        # edw aferw apo to ship_number to 1 gia na parw to epi8umhto ship index
        ship_idx = ship_number - 1

        reward_obtained, cii_attained_total, delay_in_days = self.calculate_reward(ship_idx, action).values()

        # region
        # print(f"To reward pou peirame apo to action {action} einai {reward_obtained}")
        # print(f"To accumulated cii gia to ship {ship_number} meta to action ginetai {cii_attained_total}")
        # print(f"To delay se meres apo to action einai {delay}")
        # endregion

        # xrhsimopoihse to cii_attained_after_trip gia na ananewseis to cii_attained feature[3] tou tensora tou selected ship
        # dhladh selected_ship_tensor[3] += cii_attained
        # den mporei na ginei kateu8eian me += giati o tensoras einai immutable (ftiaxnw kainourio tensora kai ton kanw concat me ton prohgoumeno)

        # xrhsimopoihse to delay gia na ananewseis tis meres pou telika tha einai unavailable to ship sto ship_log
        # etsi 8a prokuptei kai ship mask
        if action != 12:
            self.update_state(ship_idx, action, cii_attained_total, delay_in_days)
            print("ekana update to state")
            print("tsekare contract_tensor,contract_mask,ship_tensor,ship_mask")
        else:
            print("ho")

        state_dict = {
            "contracts_state": self.contracts_tensor,
            "ships_state": self.ships_tensor,
            "contracts_mask": self.contracts_mask,
            "ships_mask": self.ships_mask,
        }

        state = tf.concat()
        return state, reward_obtained, done

    def reset(self):
        """
        `reset` sets the environment to its initial state

        Returns:
        * initial_state : the initial state / observation of the environment.

        """
        np.random.seed(4)  # bgalto meta
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
        (self.contracts_df, self.contracts_tensor,) = self.create_contracts_tensor()

        # Add the ballast distances to the ships tensor
        self.ships_tensor = func_ballast(con_tensor=self.contracts_tensor, ships_tensor=self.ships_tensor, dm_tensor=self.dm_tensor,)

        # bale ta ballast distances pisw sto ships df
        self.ships.loc[:, ["ballast_1", "ballast_2", "ballast_3", "ballast_4"]] = self.ships_tensor[:, -4:].numpy()

        # An entity showing which daily contracts were taken (1) and which were not (0)
        # self.contracts_mask = tf.ones(shape=(self.NUM_DAILY_CONTRACTS, 1))
        # self.contracts_mask = self.contracts_tensor[:,7]
        self.contracts_mask = tf.convert_to_tensor(np.array([[0], [1], [0], [1]]), dtype=tf.float32)

        # An entity showing for how many days each ship is going to be unavailable due to serving a contract
        # days_of_unavailability = (balast_distance of that contract + contract_distance) / picked_speed
        # ship_log = {ship_number:days_of_unavailability}
        self.ships_log = {1: 0, 2: 0, 3: 0, 4: 0}

        #
        # to ships_mask tha pairnei 1 gia opoio ship sto ship_log exei days_of_unavailability = 0 alliws tha pairnei 0
        # arxika epeidh to ship_log exei gia ola ta keys values 0 to ships_mask exei 4 asous
        # TODO bale ena if pou na bazei 1 sto ships_mask an to ship_log[ship_number] == 0 alliws an ship_log[ship_number]!=0 na bazei 0
        self.ships_mask = tf.ones(shape=(self.NUM_SHIPS, 1))

        self.state = {
            "contracts_state": self.contracts_tensor,
            "ships_state": self.ships_tensor,
            "contracts_mask": self.contracts_mask,
            "ships_mask": self.ships_mask,
        }

        return self.state

    def create_contracts(self):
        """
        `create_contracts` creats cargo contracts for a specific day of the year
        """
        # auto bgalto meta
        np.random.seed(7)
        con_df = pd.DataFrame(
            columns=["start_port_number", "end_port_number", "contract_type", "start_day", "end_day", "cargo_size", "contract_duration", "contract_availability", "contract_distance", "value",]
        )

        ship_types = np.array(["supramax", "ultramax", "panamax", "kamsarmax"])

        con_df["start_port_number"] = np.random.randint(1, self.NUM_PORTS + 1, size=self.NUM_DAILY_CONTRACTS)
        con_df["contract_type"] = np.random.choice(ship_types, size=self.NUM_DAILY_CONTRACTS)
        con_df["end_port_number"] = np.random.randint(1, self.NUM_PORTS + 1, size=self.NUM_DAILY_CONTRACTS)

        same_ports = con_df["start_port_number"] == con_df["end_port_number"]
        # check that start and end ports are different
        while sum(same_ports) != 0:
            con_df["end_port_number"] = np.where(same_ports, np.random.randint(low=1, high=self.NUM_PORTS + 1, size=same_ports.shape,), con_df["end_port_number"],)
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

        dt_days, dt_hours = find_duration(distance=dx, u=u_picked)

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

        # add contract value : einai analogo tou (kg * miles) / time at sea
        con_df["value"] = round(con_df["cargo_size"] * (con_df["contract_distance"] / (con_df["contract_duration"] * 1_000_000)))

        # set contract availability to 1 for each contract
        con_df["contract_availability"] = np.ones(shape=(self.NUM_DAILY_CONTRACTS))

        return con_df

    def create_contracts_tensor(self):
        """
        `create_contracts_tensor` creates a tensor out of the contracts dataframe
        """
        empty = pd.DataFrame(
            columns=["start_port_number", "end_port_number", "contract_type", "start_day", "end_day", "cargo_size", "contract_duration", "contract_availability", "contract_distance", "value",]
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

    def find_trip_distance(self, ship_idx, contract, cp):

        """
        `find_trip_distance` calculates the trip distance of a ship serving a contract

        contract : selected contract
        ship_idx : ship_number - 1
        """

        # to contract distance einai to feat[8] tou selected contract tensora
        selected_contract_distance = self.contracts_tensor[contract, 8]

        #
        selected_start_port = self.contracts_tensor[contract, 0]

        selected_end_port = self.contracts_tensor[contract, 1]

        # print(f"We chose contract {contract}")
        print(f"To start port tou contract_{contract} einai to {selected_start_port}")
        print(f"To end port tou contract_{contract} einai to {selected_end_port}")
        print(f"H apostash metaksy twn dyo autwn port einai {selected_contract_distance} nm")

        # analoga me to poio contract dialeksa
        # epilegw to antistoixo ballast apo ton tensora tou selected ship
        # to briskw me contract number mod 4 + 7 pou einai to index pou ksekinane ta ballast features

        ballast_feature_idx = contract % 4 + 7
        print(f"To ballast_idx pou epileksame einai to ballast_{ballast_feature_idx-7} pou einai to feature {ballast_feature_idx} tou ship_tensor")
        print(f"To ploio {ship_idx+1} brisketai sto current port {cp}")
        selected_ballast_distance = self.ships_tensor[ship_idx, ballast_feature_idx]
        print(f"To ballast distance metaksy current port {cp} kai start port {selected_start_port} prokyptei {selected_ballast_distance} nm")
        total_distance = selected_contract_distance + selected_ballast_distance
        print(f"To synoliko distance einai {selected_contract_distance} nm + {selected_ballast_distance} nm = {total_distance} nm")

        return total_distance

    def calculate_reward(self, ship_idx, action):

        """
        `calculate_reward` calculates the reward obtained by the action for the selected ship.

        Inputs:

        * ship_idx : the index of the ship selected in the loop. ship_idx in [0,1,2,3]
        * action : the action for the selected ship

        """

        ship_number = ship_idx + 1

        # mapare to action se contract kai speed
        contract, speed = map_action(action)

        print(f"The contract selected is contract_{contract} and the speed selected for ship {ship_number} is {speed} knots")

        # an apofasises na pareis kapoio contract
        if action != 12:
            print("Bhka sto if, phra contract ! :D")

            reward_obtained = 0

            # pare ton tensora tou ship pou dialekses mesw tou loop
            selected_ship_tensor = self.ships_tensor[ship_idx, :]

            # pare to current port apo ton ship tensora
            cp = selected_ship_tensor[4]

            print(f"To reward sthn arxh einai {reward_obtained}")

            contract_value = self.contracts_tensor[contract, 9]

            # print(f"To value tou contract_{contract} einai {contract_value}")

            # total_trip_distance =  ballast + contract distance
            total_trip_distance = self.find_trip_distance(ship_idx, contract, cp)
            # print(f"To total distance tou trip (ballast + contract distance) einai {total_trip_distance} nm")

            # briskw to duration tou trip se meres kai wres
            trip_duration_days, trip_duration_hours = find_duration(u=speed, distance=total_trip_distance,)
            # print(f"To duration tou trip (ballast + contract distance) se hours einai {trip_duration_hours}")
            # print(f"To duration tou trip (ballast + contract distance) se days einai {trip_duration_days}")

            # delay= actual_trip_duration - contract_duration
            delay = trip_duration_days - self.contracts_tensor[contract, 6]
            if delay > 0:
                print(f" To ship {ship_number} 8a arghsei {delay} meres")
            elif delay == 0:
                print(f"To ship {ship_number} den tha arghsei na")
            else:
                print(f"To ship {ship_number} 8a ftasei {-delay} meres nwritera")

            # an arghsa kapoies meres (delay>0) plhrwse kostos 10 monades gia ka8e mera argoporias
            # alliws krata to thetiko reward (-delay) epeidh eftases sthn wra sou
            delay_cost_per_day = -10
            delay_reward = delay * delay_cost_per_day if delay > 0 else -delay

            # cii = cii_threshold - cii_attained
            cii_threshold = selected_ship_tensor[2]

            # to accumulated cii_attained tou ship mexri twra
            accumulated_cii_attained_till_now = selected_ship_tensor[3]
            # print(f"The cii threshold of the selected ship {ship_number} is {cii_threshold}")

            # to cii pou parax8hke apo to twrino trip
            cii_attained_current_trip = find_cii_attained(ship_number=ship_number, speed=speed, distance=total_trip_distance,)

            # print(f"The attained cii for the selected ship {ship_number} during the current trip is {cii_attained_current_trip}")

            # add the cii attained in the current trip to the cii attained until now
            accumulated_cii_attained_after_trip = accumulated_cii_attained_till_now + cii_attained_current_trip

            # h diafora tou cii_threshold apo to accumulated cii
            cii_difference = cii_threshold - accumulated_cii_attained_after_trip

            # an cii_difference > 0 dhladh to ship exei akoma peri8wrio oso afora to cii_threshold tote
            # dwse cii_reward = cii_difference alliws
            # an cii_difference < 0 dwse cii_reward = 1000 * cii_difference

            # cii_cost an kseperasw to cii_threshold
            cii_cost = 1000

            # to reward apo to cii
            cii_reward = cii_difference if cii_difference > 0 else cii_difference * cii_cost
            # print(f"The reward component regarding the cii is {cii_reward} ")

            reward_obtained = contract_value + cii_reward + delay_reward

            print(f"The total reward is made up by the contract value {contract_value}")
            print(f"The delay reward is {delay_reward}")
            print(f"The cii reward is {cii_reward}")

        else:
            # TODO allakse thn calculated_rewards gia na fernei pisw mono to reward oxi ta delays klp
            print("Bhka sto else, eimai no take :(")
            # an den phra conract

            # region

            # # pare ton tensora tou ship pou dialekses mesw tou loop
            selected_ship_tensor = self.ships_tensor[ship_idx, :]

            # # to cii_attained tou ship mexri twra
            accumulated_cii_attained_till_now = selected_ship_tensor[3]

            # # to accumulated cii_attained_after_trip praktika tha einai to idio me prin to accumulated cii_attained_till_now + 0
            accumulated_cii_attained_after_trip = accumulated_cii_attained_till_now + 0

            # endregion

            # to delay balto 0

            delay = 0

            # bale ena mikro arnhtiko reward epeidh den phres tipota
            reward_obtained = -1

        # eite pareis contract eite oxi orise to reward_dict
        reward_dict = {"reward_obtained": reward_obtained, "accumulated_cii_attained_after_trip": accumulated_cii_attained_after_trip, "delay_in_days": delay}

        return reward_dict

    def update_state(self, ship_idx, action, cii_attained, delay):

        """
        `update_state` updates the environment state based on the `action` taken
        """
        ship_number = ship_idx + 1

        # an phres contract
        if action != 12:

            # kane map to action gia na breis poio contract kai poio speed par8hke
            contract, speed = map_action(action)

            # region contracts state

            contract_updates = self.update_contract_tensor(contract)

            contract_updates = tf.reshape(contract_updates, shape=(1, contract_updates.shape[0]))

            indices_con = tf.constant([[contract]])

            contract_tensors = self.contracts_tensor

            self.contracts_tensor = tf.tensor_scatter_nd_update(contract_tensors, indices_con, contract_updates)

            # endregion

            # region contracts mask

            # to contracts mask ginetai h sthlh contract_availability tou contracts tensora

            self.contracts_mask = self.contracts_tensor[:, 7]

            # endregion

            # region ships state
            cii_attained_total = cii_attained
            print(cii_attained_total)

            ship_updates = self.update_ship_tensor(ship_idx, contract, speed, cii_attained_total)

            ship_updates = tf.reshape(ship_updates, shape=(1, ship_updates.shape[0]))

            indices_ship = tf.constant([[ship_idx]])

            ships_tensors = self.ships_tensor

            self.ships_tensor = tf.tensor_scatter_nd_update(ships_tensors, indices_ship, ship_updates)
            # endregion

            # region ships_mask

            # to contract mask ginetai h sthlh ship_availability tou ships tensora
            self.ships_mask = self.ships_tensor[:, 6]
            # endregion

            # region ships_log

            self.ships_log[ship_number] = delay

            # endregion

        else:
            # an den phres contract action==12

            # to state den allazei
            print("Den phra kapoio contract opote nothing changed")

        pass

    def update_contract_tensor(self, contract):
        """
        `update_contract_tensor` updates the `contract_availability` feature of the selected contract to 0
        to signal that this contract became unavailable

        Inputs:

        * `contract`: the selected contract

        Outputs:

        * `output`: the tensor of the selected contract with 1 in the `contract_availability` field
        """

        # pare ton tensora tou selected contract
        selected_contract_tensor = tf.cast(self.contracts_tensor[contract, :], tf.float32)

        # thelw na balw 0 sto contract_availability
        inplace_array = np.array([0])

        # ftiaxnw enan array me 0 kai 1 stis 8eseis pou 8elw na allaksw (edw h thesh 7 tou contract_availability feature)
        # indices_array = np.array([0,0,0,0,0,0,0,1,0,0])
        indices_array = np.zeros(10)
        indices_array[7] = 1

        # pairnw to index tou feature pou thelw na allaksw se morfh tensora (contract_availability)
        indices = tf.cast(tf.where(tf.equal(indices_array, 1)), tf.int32)

        # ftiaxnw enan scattered array me tis inplace times (1) sta antistoixa feature indices pou thelw na allaksw (7)
        scatter = tf.scatter_nd(indices, inplace_array, shape=tf.shape(selected_contract_tensor))

        # o inverse_mask einai o logikos antistrofos tou indices_array
        inverse_mask = tf.cast(tf.math.logical_not(indices_array), tf.float32)

        # selected_contract_tensor me contract_availability na einai 1
        input_array_zero_out = tf.multiply(inverse_mask, selected_contract_tensor)

        # selected_contract_tensor me contract_availability na einai 0
        output = tf.add(input_array_zero_out, tf.cast(scatter, tf.float32))

        output = tf.cast(output, tf.float32)

        return output

    def update_ship_tensor(self, ship_idx, contract, speed, cii_attained):
        """

        """

        # pare ton tensora tou selected ship
        selected_ship_tensor = tf.cast(self.ships_tensor[ship_idx, :], tf.float32)

        # pare ton tensora tou selected contract
        selected_contract_tensor = self.contracts_tensor[contract, :]

        # thelw na balw tis ekshs times
        ship_availability = 0
        new_current_port = selected_contract_tensor[1].numpy()
        current_speed = speed
        new_cii_attained = cii_attained

        #
        inplace_array = np.array([new_cii_attained, new_current_port, current_speed, ship_availability])

        # ftiaxnw enan array me 0 kai 1 stis 8eseis pou 8elw na allaksw 8eseis [3,4,5,6]
        # indices_array = np.array([0,0,0,0,0,0,0,1,0,0])
        indices_array = np.zeros(11)
        indices_array[[3, 4, 5, 6]] = 1

        # pairnw ta indices twn features pou thelw na allaksw se morfh tensora (cii_attained,current_port,current_speed,ship_availability)
        indices = tf.cast(tf.where(tf.equal(indices_array, 1)), tf.int32)

        # ftiaxnw enan scattered array me tis inplace times sta antistoixa feature indices pou thelw na allaksw
        scatter = tf.scatter_nd(indices, inplace_array, shape=tf.shape(selected_ship_tensor))

        # o inverse_mask einai o logikos antistrofos tou indices_array
        inverse_mask = tf.cast(tf.math.logical_not(indices_array), tf.float32)

        #
        input_array_zero_out = tf.multiply(inverse_mask, selected_ship_tensor)

        # o ananewmenos selected ship tensor me tis times pou 8elw
        output = tf.add(input_array_zero_out, tf.cast(scatter, tf.float32))

        output = tf.cast(output, tf.float32)

        return output
