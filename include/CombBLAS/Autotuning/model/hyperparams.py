
class HyperParams:

    def __init__(self, infile):
        
        params_dict = {}

        with open(infile, 'r') as file:
            for line in file:
                param, val = line.split(":")[0], line.split(":")[1]
                params_dict[param] = float(val)

        for param in params_dict:
            setattr(self, param, params_dict[param])


