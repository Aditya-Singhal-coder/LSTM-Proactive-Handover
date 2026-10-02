class BaseStation:
    def __init__(self, bs_id, x, y):
        self.bs_id = bs_id
        self.x = x
        self.y = y

    def position(self):
        return self.x, self.y


def create_base_stations():
    return [
        BaseStation("BS1", 0, 0),
        BaseStation("BS2", 500, 0),
        BaseStation("BS3", 250, 433)
    ]