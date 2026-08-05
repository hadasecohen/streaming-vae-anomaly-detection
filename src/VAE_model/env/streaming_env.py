
class StreamEnv:
    def __init__(
            self,
            device,
            #trainer: StreamTrainer,  # configured with model/opt/loss/policy
        ):
        
        self.device = device
        self.run_files_output = None
    
        # self.trainer = trainer  # configured with model/opt/loss/policy

    def do_something(self, x):
        raise NotImplementedError("Subclasses must implement do_something()")

    def close_env(self) :
        self.run_files_output.close()
    