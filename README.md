# Reinforcement learning for civilization like game

# Game
The developed environment consists of the following elements:
- 5x5 square grid area
- Agent begins the game with 10 gold
- From the first turn, the agent has two units at their disposal: settler and scout.
- Settler:
  - by default, it appears on the (0, 0) square
  - moves a maximum of 1 square per turn
  - can build cities
  - can collect resources
- Scout:
  - by default, it appears on the (1, 1) square
  - moves a maximum of 2 squares per turn
  - can collect resources
- Units can move in the directions of up, down, left, right.
- Three types of resources: gold, wood, iron. Random amounts of these resources appear on random fields on the map in the first turn. The total of all resources equals 5.
- Three types of cities that can be built: basic, wooden, iron.
- Basic City:
  - building cost: 5 pieces of gold.
  - passively generates 1 piece of gold per turn.
- Wooden City:
  - building cost: 7 pieces of gold, 1 piece of wood.
  - passively generates 2 piece of gold per turn.
- Iron City:
  - building cost: 10 pieces of gold, 1 piece of wood, 1 piece of iron.
  - passively generates 3 piece of gold per turn.
- Each game lasts for 25 turns.


# Reinforcement learning
Q-learning algorithm with reward values for maximization of gold amount at the end of 25th turn. 

Example results for α	= 0.075, γ = 0.99, ε = 1.0, ε_decay = 0.0005, epochs = 100 000
Visualization of last training epoch:

![game_visualization](https://github.com/plmrr/reinforcement-learning/assets/130595899/1a1c6408-7b56-4749-b776-72e09e9aeeea)


Train results:

![gold_plot](https://github.com/plmrr/reinforcement-learning/assets/130595899/a4a315ef-747e-4bb2-bc93-c99b1deea95e)

![reward_plot](https://github.com/plmrr/reinforcement-learning/assets/130595899/45f6baf7-78ef-4fc6-9abe-910967b19c92)
