class Explainer:
    def __init__(self):
        self.sys_prompt = None

    def _initialise_model(self):
        raise NotImplementedError

    def _create_prompt(self):
        raise NotImplementedError

    def set_sys_prompt(self, sys_prompt):
        self.sys_prompt = sys_prompt

    def explain(self):
        raise NotImplementedError
    