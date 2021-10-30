import gym
from utils.utils import cii_expected

class CarbonEnv(gym.Env):
    """
    Description :

        A custom openai gym environment for the carbon emission problem.

    """


    def __init__(self,data_dict):
        super().__init__()

        NUM_CONTRACTS_PER_STEP = 4
        NUM_SHIPS = 10
        SET_OF_SPEEDS = [10,12,14]
        MIN_REVENUE = 1_000_000
       # TERMINATION_FROM_CII = # AT LEAST 5 SHIPS DROP THEIR RATING

        self.fleet = pd.read_csv(data_dict['fleet_path'])
        self.ports = pd.read_csv(data_dict['ports_path'])
        self.dm = pd.read_csv(data_dict['dm_path'])

        # calculate fleet's required cii
        self.fleet['cii_required'] = self.fleet['dwt'].map(cii_expected)

        # set fleet at random ports
        self.fleet['current_port'] = np.random.randint(1,11,self.fleet.shape[0])

        self.state = None
        self.previous_action = None



        observation_space = spaces.Dict({
            "daily_contracts" : spaces.Discrete(NUM_CONTRACTS_PER_STEP,), # 4 new contracts each day
            "available_ships" : spaces.Box(low=0,high=1,shape=(NUM_SHIPS,),dtype=np.int32) # [0,1,1,0,0,1,0,1,1,0]
                                                                                            # ships: 1,2,5,7,8 are available
        })

        action_space = spaces.Dict({
            "choose_contract" : spaces.Box(low=0,high=1,shape=(4,),dtype=np.int32), # (0,0,1,1) deny contracts 0,1 accept contracts 2,3
            "choose_ship" : spaces.Box(low=0, high=10, shape=(4,),dtype=np.int32), # (0,0,5,7) assign ships 5,7 to contracts 2,3
            "choose_speed" : spaces.Box(low=0, high=3, shape=(4,),dtype=np.int32) # (0,0,1,2) 10 knots for ship 5, 12knots for ship 7  
            })


        self.reset()

    def create_contracts(self, seed=None):
        """
        A method that creates 4 random cargo contracts at each env step. \ 
        There are 5 cargo contract types handy,supra,ultra,pana,kamsar for each port.\ 
        The function :
        * returns a 1d array of 50 elements
        * all array elements are integers from 0 to 4
        * the sum of the array must be 4
        """
        contract_count = 4
        np.random.seed(seed)
        x = np.random.multinomial(contract_count, [1/5]*5, None)
        z = np.zeros(shape=45, dtype=np.int32)
        c = np.concatenate((x,z))
        np.random.shuffle(c)
        return c




    def step(self, action):
        # Execute one time step in the environment based on the action chosen by the model
        # action will have shape 3x4
        print(len(action))
        print(action)
        pass

    def reset(self):
        # Set the state of the environment to an initial state
        # return self._RESET()
        pass

    def _STEP(self,action_dict):
        pass
        
    def _RESET(self):
        #return self.state
        pass





    pass


