import random


class Unit:
    def __init__(self, position):
        self.position = position


class Settler(Unit):
    def __init__(self, position, steps_per_turn=1):
        super().__init__(position)
        self.steps_per_turn = 1


class Scout(Unit):
    def __init__(self, position, steps_per_turn=2):
        super().__init__(position)
        self.steps_per_turn = 2


class Game:
    def __init__(self, board_size=5, num_resources=5):
        self.turn = 0
        self.board_size = board_size
        self.num_resources = num_resources
        self.units = []
        self.gold = 10
        self.wood = 0
        self.iron = 0
        self.resource_positions = self.generate_resources(num_resources)
        self.city_positions = []
        self.wood_city_positions = []
        self.iron_city_positions = []
        self.city_cost = {'gold': 5, 'wood': 0, 'iron': 0}
        self.wood_city_cost = {'gold': 7, 'wood': 1, 'iron': 0}
        self.iron_city_cost = {'gold': 10, 'wood': 1, 'iron': 1}
        self.city_income = 1
        self.wood_city_income = 2
        self.iron_city_income = 3

    def generate_resources(self, num_resources):
        resource_positions = {}
        for _ in range(num_resources):
            while True:
                x = random.randint(0, self.board_size - 1)
                y = random.randint(0, self.board_size - 1)
                if (x, y) not in resource_positions:
                    resource_positions[(x, y)] = random.choice(['wood', 'iron', 'gold'])
                    break
        return resource_positions

    def add_unit(self, unit_type, position):
        if unit_type == 'settler':
            self.units.append(Settler(position))
        elif unit_type == 'scout':
            self.units.append(Scout(position))

        for unit in self.units:
            if unit.position in self.resource_positions:
                resource = self.resource_positions[unit.position]
                if resource == 'wood':
                    self.wood += 1
                elif resource == 'iron':
                    self.iron += 1
                elif resource == 'gold':
                    self.gold += 1
                del self.resource_positions[unit.position]

    def move_unit(self, unit, direction, steps):
        unit = self.units[unit]
        for _ in range(min(steps, unit.steps_per_turn)):
            if direction == 'up':
                unit.position = (max(0, unit.position[0] - 1), unit.position[1])
            elif direction == 'down':
                unit.position = (min(self.board_size - 1, unit.position[0] + 1), unit.position[1])
            elif direction == 'left':
                unit.position = (unit.position[0], max(0, unit.position[1] - 1))
            elif direction == 'right':
                unit.position = (unit.position[0], min(self.board_size - 1, unit.position[1] + 1))

            if unit.position in self.resource_positions:
                resource = self.resource_positions[unit.position]
                if resource == 'wood':
                    self.wood += 1
                elif resource == 'iron':
                    self.iron += 1
                elif resource == 'gold':
                    self.gold += 1
                del self.resource_positions[unit.position]

    def build_city(self, city_type, unit):
        unit = self.units[unit]
        if city_type == 'basic':
            if self.gold >= self.city_cost['gold'] and self.wood >= self.city_cost['wood'] and self.iron >= \
                    self.city_cost['iron']:
                self.city_positions.append(unit.position)
                self.gold -= self.city_cost['gold']
                self.wood -= self.city_cost['wood']
                self.iron -= self.city_cost['iron']
        elif city_type == 'wood':
            if self.gold >= self.wood_city_cost['gold'] and self.wood >= self.wood_city_cost['wood'] and self.iron >= \
                    self.wood_city_cost['iron']:
                self.wood_city_positions.append(unit.position)
                self.gold -= self.wood_city_cost['gold']
                self.wood -= self.wood_city_cost['wood']
                self.iron -= self.wood_city_cost['iron']
        elif city_type == 'iron':
            if self.gold >= self.iron_city_cost['gold'] and self.wood >= self.iron_city_cost['wood'] and self.iron >= \
                    self.iron_city_cost['iron']:
                self.iron_city_positions.append(unit.position)
                self.gold -= self.iron_city_cost['gold']
                self.wood -= self.iron_city_cost['wood']
                self.iron -= self.iron_city_cost['iron']

    def collect_income(self):
        self.gold += len(self.city_positions) * self.city_income
        self.gold += len(self.wood_city_positions) * self.wood_city_income
        self.gold += len(self.iron_city_positions) * self.iron_city_income

    def print_board(self):
        board = [[' ' for _ in range(self.board_size)] for _ in range(self.board_size)]

        for position, resource in self.resource_positions.items():
            if resource == 'wood':
                board[position[0]][position[1]] = 'W'
            elif resource == 'iron':
                board[position[0]][position[1]] = 'I'
            elif resource == 'gold':
                board[position[0]][position[1]] = 'G'

        for position in self.city_positions:
            board[position[0]][position[1]] = 'C'
        for position in self.wood_city_positions:
            board[position[0]][position[1]] = 'WC'
        for position in self.iron_city_positions:
            board[position[0]][position[1]] = 'IC'

        for unit in self.units:
            if isinstance(unit, Settler):
                board[unit.position[0]][unit.position[1]] = 'S'
            elif isinstance(unit, Scout):
                board[unit.position[0]][unit.position[1]] = 'SC'

        for row in board:
            print(' '.join(row))

    def get_state(self):
        state = {
            'units': [(isinstance(unit, Settler), unit.position) for unit in self.units],
            'basic_cities': self.city_positions,
            'wooden_cities': self.wood_city_positions,
            'iron_cities': self.iron_city_positions,
            'resources': self.resource_positions,
            'gold': self.gold,
            'wood': self.wood,
            'iron': self.iron
        }
        return state

    def reset(self):
        self.units = []
        self.turn = 0
        self.resource_positions = self.generate_resources(self.num_resources)
        self.add_unit('settler', (0, 0))
        self.add_unit('scout', (1, 1))
        self.gold = 10
        self.wood = 0
        self.iron = 0
        self.city_positions = []
        self.wood_city_positions = []
        self.iron_city_positions = []
        return self.get_state()

    def step(self, action):
        unit_id, action_type, direction, steps = action
        self.turn += 1

        if self.turn == 25:
            self.collect_income()
            return self.get_state(), 0, True

        if action_type == 0:  # move
            if direction == 0:  # up
                if (self.units[unit_id].position[1] - steps) < 0:
                    self.collect_income()
                    return self.get_state(), -99, False
            elif direction == 1:  # down
                if (self.units[unit_id].position[1] + steps) >= self.board_size:
                    self.collect_income()
                    return self.get_state(), -99, False
            elif direction == 2:  # right
                if (self.units[unit_id].position[0] + steps) >= self.board_size:
                    self.collect_income()
                    return self.get_state(), -99, False
            elif direction == 3:  # left
                if (self.units[unit_id].position[0] - steps) < 0:
                    self.collect_income()
                    return self.get_state(), -99, False
            old_position = self.units[unit_id].position
            self.move_unit(unit_id, direction, steps)
            new_position = self.units[unit_id].position
            if new_position != old_position:
                if new_position in self.resource_positions:
                    resource = self.resource_positions[new_position]
                    if resource == 'wood':
                        self.wood += 1
                    elif resource == 'iron':
                        self.iron += 1
                    elif resource == 'gold':
                        self.gold += 1
                    del self.resource_positions[new_position]
                    self.collect_income()
                    return self.get_state(), 2, False
            self.collect_income()
            return self.get_state(), 0, False
        elif action_type == 1:  # build
            if direction == 0:  # basic
                if self.gold >= 5 and self.units[unit_id].position not in self.city_positions + self.wood_city_positions + self.iron_city_positions:
                    self.build_city('basic', unit_id)
                    self.collect_income()
                    return self.get_state(), 15, False
                else:
                    self.collect_income()
                    return self.get_state(), -10, False
            elif direction == 1:  # wood
                if self.gold >= 7 and self.wood >= 1 and self.units[unit_id].position not in self.city_positions + self.wood_city_positions + self.iron_city_positions:
                    self.build_city('wood', unit_id)
                    self.collect_income()
                    return self.get_state(), 30, False
                else:
                    self.collect_income()
                    return self.get_state(), -10, False
            elif direction == 2:  # iron
                if self.gold >= 10 and self.wood >= 1 and self.iron >= 1 and self.units[unit_id].position not in self.city_positions + self.wood_city_positions + self.iron_city_positions:
                    self.build_city('iron', unit_id)
                    self.collect_income()
                    return self.get_state(), 50, False
                else:
                    self.collect_income()
                    return self.get_state(), -10, False
        self.collect_income()
        return self.get_state(), 0, False

