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
        self.wood = 1
        self.iron = 0
        self.resource_positions = self.generate_resources(num_resources)
        self.city_positions = []
        self.wood_city_positions = []
        self.iron_city_positions = []
        self.city_cost = {'gold': 5, 'wood': 0, 'iron': 0}
        self.wood_city_cost = {'gold': 7, 'wood': 1, 'iron': 0}
        self.iron_city_cost = {'gold': 10, 'wood': 2, 'iron': 1}
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
            'cities': self.city_positions + self.wood_city_positions + self.iron_city_positions,
            'resources': self.resource_positions,
            'gold': self.gold,
            'wood': self.wood,
            'iron': self.iron
        }
        return state

    def reset(self):
        self.units = []
        self.turn = 0
        self.add_unit('settler', (0, 0))
        self.add_unit('scout', (1, 1))
        self.gold = 10
        self.wood = 0
        self.iron = 0
        self.resource_positions = self.generate_resources(self.num_resources)
        self.city_positions = []
        self.wood_city_positions = []
        self.iron_city_positions = []
        return self.get_state()

    def step(self, action):
        unit_id, action_type, direction, steps = action
        self.turn += 1

        if self.turn == 25:
            return self.get_state(), 0, True

        if action_type == "move":
            unit = self.units[unit_id]
            unit_x = unit.position[0]
            unit_y = unit.position[1]

            if direction == 'up':
                if (unit_y - steps) <= -1:
                    return self.get_state(), -1, False
            elif direction == 'down':
                if (unit_y + steps) >= 5:
                    return self.get_state(), -1, False
            elif direction == 'right':
                if (unit_x + steps) >= 5:
                    return self.get_state(), -1, False
            elif direction == 'left':
                if (unit_x - steps) <= -1:
                    return self.get_state(), -1, False

            gold_before = self.gold
            wood_before = self.wood
            iron_before = self.iron
            self.move_unit(unit_id, direction, steps)
            if self.gold > gold_before or self.wood > wood_before or self.iron > iron_before:
                return self.get_state(), 1, False
        elif action_type == "build":
            #             if direction == 'settler':
            #                 if self.gold >= 5:
            #                     self.add_unit('settler', self.units[unit_id].position)
            #                     self.gold -= 5
            #             elif direction == 'scout':
            #                 if self.gold >= 7:
            #                     self.add_unit('scout', self.units[unit_id].position)
            #                     self.gold -= 7
            unit = self.units[unit_id]
            empty_position = unit.position not in self.city_positions + self.wood_city_positions + self.iron_city_positions
            if direction == 'basic':
                if self.gold >= 5 and empty_position and (isinstance(unit, Settler)):
                    self.build_city(direction, unit_id)
                    return self.get_state(), 1, False
                else:
                    return self.get_state(), -1, False
            elif direction == 'wood':
                if self.gold >= 7 and self.wood >= 1 and empty_position and (isinstance(unit, Settler)):
                    self.build_city(direction, unit_id)
                    return self.get_state(), 2, False
                else:
                    return self.get_state(), -1, False
            elif direction == 'iron':
                if self.gold >= 10 and self.wood >= 2 and self.iron >= 1 and empty_position and (isinstance(unit, Settler)):
                    self.build_city(direction, unit_id)
                    return self.get_state(), 3, False
                else:
                    return self.get_state(), -1, False
        self.collect_income()
        return self.get_state(), 0, False
