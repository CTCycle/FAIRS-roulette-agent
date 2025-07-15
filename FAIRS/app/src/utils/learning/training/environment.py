import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from FAIRS.app.src.utils.data.process.mapping import RouletteMapper
from FAIRS.app.src.constants import CONFIG, STATES, NUMBERS, PAD_VALUE
from FAIRS.app.src.logger import logger



# [ROULETTE RL ENVIRONMENT]
###############################################################################
class BetsAndRewards:

    def __init__(self, configuration):
        self.seed = configuration["SEED"]         
        self.bet_amount = configuration["environment"]["BET_AMOUNT"]

        mapper = RouletteMapper(configuration)   
        self.numbers = list(range(NUMBERS)) 
        self.red_numbers = mapper.color_map['red']
        self.black_numbers = mapper.color_map['black'] 

    #--------------------------------------------------------------------------
    # Define the action space with additional bets:
    # 0-36: Bet on a specific number,
    # 37: Bet on Red,
    # 38: Bet on Black,
    # 39: Pass,
    # 40: Bet on Odd,
    # 41: Bet on Even,
    # 42: Bet on Low (1-18),
    # 43: Bet on High (19-36),
    # 44: Bet on First Dozen (1-12),
    # 45: Bet on Second Dozen (13-24),
    # 46: Bet on Third Dozen (25-36)
    #--------------------------------------------------------------------------
    def bet_on_number(self, action: int, next_extraction: int):       
        if action == next_extraction:
            reward = 35 * self.bet_amount
        else:
            reward = -self.bet_amount

        return reward, False

    #--------------------------------------------------------------------------
    def bet_on_red(self, next_extraction: int):     
        if next_extraction in self.red_numbers:
            reward = self.bet_amount
        else:
            reward = -self.bet_amount

        return reward, False

    #--------------------------------------------------------------------------
    def bet_on_black(self, next_extraction: int):       
        if next_extraction in self.black_numbers:
            reward = self.bet_amount
        else:
            reward = -self.bet_amount

        return reward, False
    
    #--------------------------------------------------------------------------
    def bet_on_odd(self, next_extraction: int):        
        if next_extraction != 0 and next_extraction % 2 == 1:
            reward = self.bet_amount
        else:
            reward = -self.bet_amount

        return reward, False

    #--------------------------------------------------------------------------
    def bet_on_even(self, next_extraction: int):        
        if next_extraction != 0 and next_extraction % 2 == 0:
            reward = self.bet_amount
        else:
            reward = -self.bet_amount
        return reward, False

    #--------------------------------------------------------------------------
    def bet_on_low(self, next_extraction: int):        
        if 1 <= next_extraction <= 18:
            reward = self.bet_amount
        else:
            reward = -self.bet_amount

        return reward, False

    #--------------------------------------------------------------------------
    def bet_on_high(self, next_extraction: int):        
        if 19 <= next_extraction <= (NUMBERS - 1):
            reward = self.bet_amount
        else:
            reward = -self.bet_amount

        return reward, False

    #--------------------------------------------------------------------------
    def bet_on_first_dozen(self, next_extraction: int):       
        if 1 <= next_extraction <= 12:
            reward = 2 * self.bet_amount
        else:
            reward = -self.bet_amount

        return reward, False

    #--------------------------------------------------------------------------
    def bet_on_second_dozen(self, next_extraction: int):        
        if 13 <= next_extraction <= 24:
            reward = 2 * self.bet_amount
        else:
            reward = -self.bet_amount

        return reward, False

    #--------------------------------------------------------------------------
    def bet_on_third_dozen(self, next_extraction: int):        
        if 25 <= next_extraction <= (NUMBERS - 1):
            reward = 2 * self.bet_amount
        else:
            reward = -self.bet_amount
        return reward, False
    
    #--------------------------------------------------------------------------
    def pass_turn(self):
        reward = 0       
        return reward, True

    #--------------------------------------------------------------------------
    def interact_and_get_rewards(self, action: int, next_extraction: int, capital : int):       
        done = False
        if 0 <= action <= 36:
            reward, done = self.bet_on_number(action, next_extraction)
        elif action == 37:
            reward, done = self.bet_on_red(next_extraction)
        elif action == 38:
            reward, done = self.bet_on_black(next_extraction)        
        elif action == 39:
            reward, done = self.bet_on_odd(next_extraction)
        elif action == 40:
            reward, done = self.bet_on_even(next_extraction)
        elif action == 41:
            reward, done = self.bet_on_low(next_extraction)
        elif action == 42:
            reward, done = self.bet_on_high(next_extraction)
        elif action == 43:
            reward, done = self.bet_on_first_dozen(next_extraction)
        elif action == 44:
            reward, done = self.bet_on_second_dozen(next_extraction)
        elif action == 45:
            reward, done = self.bet_on_third_dozen(next_extraction)
        elif action == 46:
            reward, done = self.pass_turn()
        
        capital += reward

        return reward, capital, done       

    
# [ROULETTE RL ENVIRONMENT]
###############################################################################
class RouletteEnvironment(gym.Env):

    def __init__(self, data, configuration):
        super(RouletteEnvironment, self).__init__() 
        self.timeseries = data['timeseries'].values
        self.positions = data['position'].values
        self.colors = data['color_code'].values
               
        self.perceptive_size = configuration["model"]["PERCEPTIVE_FIELD"]        
        self.initial_capital = configuration["environment"]["INITIAL_CAPITAL"]
        self.bet_amount = configuration["environment"]["BET_AMOUNT"]
        self.max_steps = configuration["environment"]["MAX_STEPS"] 
        self.render_environment = configuration["environment"]["RENDERING"]

        self.player = BetsAndRewards(configuration)  

        self.black_numbers = self.player.black_numbers
        self.red_numbers = self.player.red_numbers
        self.odd_numbers = [n for n in range(1, NUMBERS) if n % 2 != 0]
        self.even_numbers = [n for n in range(1, NUMBERS) if n % 2 == 0]
        self.low_numbers = list(range(1, 19))       
        self.high_numbers = list(range(19, NUMBERS))  
        self.first_dozen_numbers = list(range(1, 13)) 
        self.second_dozen_numbers = list(range(13, 25))
        self.third_dozen_numbers = list(range(25, NUMBERS))               
                
        # Actions: 0 (Red), 1 (Black), 2-37 for betting on a specific number
        self.numbers = list(range(NUMBERS))  
        self.action_space = spaces.Discrete(STATES)
        # Observation space is the last perceptive_field numbers that appeared on the wheel
        self.observation_space = spaces.Box(
            low=0, high=36, shape=(self.perceptive_size,), dtype=np.int32)
        
        # Initialize state, capital, steps, and reward  
        self.extraction_index = 0 
        self.state = np.full(shape=self.perceptive_size, fill_value=PAD_VALUE)                       
        self.capital = self.initial_capital
        self.steps = 0
        self.reward = 0
        self.done = False
        
        if self.render_environment:
            self._build_rendering_canvas()            
        
    # Reset the state of the environment to an initial state
    #--------------------------------------------------------------------------
    def reset(self, start_over=False):        
        self.extraction_index = 0 if start_over else self.select_random_index() 
        self.state = np.full(
            shape=self.perceptive_size, fill_value=PAD_VALUE, dtype=np.int32)                  
        self.capital = self.initial_capital
        self.steps = 0
        self.done = False

        return self.state
    
    # Reset the state of the environment to an initial state
    #--------------------------------------------------------------------------
    def scale_rewards(self, rewards): 
        # Scale negative rewards to [-1, 0] and positive rewards to [0, 1]
        negative_scaled = (
            (rewards - (-self.bet_amount)) / (0 - (-self.bet_amount))) * (0 - (-1)) + (-1)        
        positive_scaled = (
            (rewards - 0) / (self.bet_amount * 35)) * (1 - 0) + 0        
        scaled_rewards = np.where(rewards < 0, negative_scaled, positive_scaled)

        return scaled_rewards
    
    # Perform the action (0: Bet on Red, 1: Bet on Black, 2: Bet on Specific Number)
    #--------------------------------------------------------------------------
    def select_random_index(self):
        end_cutoff = self.timeseries.shape[0] - self.perceptive_size
        random_index = np.random.randint(0, end_cutoff)

        return random_index

    # Perform the action (0: Bet on Red, 1: Bet on Black, 2: Bet on Specific Number)
    #--------------------------------------------------------------------------
    def update_rewards(self, action, next_extraction):
        self.reward, self.capital, self.done = self.player.interact_and_get_rewards(
                action, next_extraction, self.capital)        
   
    #--------------------------------------------------------------------------
    def step(self, action):  
        # reset the perceived field each time the end of the series is reached
        # then start again from a random index simulating a brand new roulette series      
        if self.extraction_index >= self.timeseries.shape[0]:
            self.state = self.reset()
        
        next_extraction = np.int32(self.timeseries[self.extraction_index])        
        self.state = np.delete(self.state, 0)
        self.state = np.append(self.state, next_extraction)
        self.extraction_index += 1

        self.update_rewards(action, next_extraction)
        self.steps += 1

        # Check if the episode should end
        if self.capital <= 0 or self.steps >= self.max_steps:
            self.done = True
        else:
            self.done = False

        return self.state, self.reward, self.done, next_extraction    

    #--------------------------------------------------------------------------
    def _build_rendering_canvas(self):
        plt.ion()  # Turn on interactive mode
        self.fig, self.ax = plt.subplots(figsize=(10,10), subplot_kw={'projection': 'polar'})
        self.fig.canvas.manager.set_window_title('Roulette Wheel')  # Set window title
        plt.show(block=False)  
        # Store references to text objects for updating
        self.title_text = self.ax.set_title("Roulette Wheel - Current Spin")
        self.episode_text = self.fig.text(0.5, 0.08, "", ha="center", fontsize=12)
        self.capital_text = self.fig.text(0.5, 0.05, "", ha="center", fontsize=12)
        self.extraction_text = self.fig.text(0.5, 0.02, "", ha="center", fontsize=10)    

    # Render the environment to the screen 
    #--------------------------------------------------------------------------
    def render(self, episode, time_step, action, extracted_number):
        self.ax.clear()
        # Assigning colors to each number to create the roulette layout
        colors = ['green'] + ['red', 'black'] * 18
        labels = list(range(NUMBERS))
        # create equal slices of the wheel to get 36 different sections
        theta = np.linspace(0, 2 * np.pi, NUMBERS, endpoint=False)
        width = 2 * np.pi / NUMBERS
        # Create edges for the drawn slices
        bars = self.ax.bar(
            theta, np.ones(NUMBERS), width=width, color=colors, edgecolor='white', align='edge')

        # Highlight the action and related bars
        highlight_color = 'blue'
        highlight_alpha = 0.7
        if 0 <= action <= 36:  # Bet on a specific number
            if action < len(bars): # Safety check
                bars[action].set_facecolor(highlight_color)
                bars[action].set_alpha(highlight_alpha)

        elif action == 37:  # Bet on Red
            for red_number in self.red_numbers:
                if red_number < len(bars): # Safety check
                    bars[red_number].set_facecolor(highlight_color)
                    bars[red_number].set_alpha(highlight_alpha)

        elif action == 38:  # Bet on Black
            for black_number in self.black_numbers:
                if black_number < len(bars): # Safety check
                    bars[black_number].set_facecolor(highlight_color)
                    bars[black_number].set_alpha(highlight_alpha)

        elif action == 40:  # Bet on Odd
            for odd_number in self.odd_numbers:
                if odd_number < len(bars): # Safety check
                    bars[odd_number].set_facecolor(highlight_color)
                    bars[odd_number].set_alpha(highlight_alpha)

        elif action == 41:  # Bet on Even
            for even_number in self.even_numbers:
                if even_number < len(bars): # Safety check
                    bars[even_number].set_facecolor(highlight_color)
                    bars[even_number].set_alpha(highlight_alpha)

        elif action == 42:  # Bet on Low (1-18)
            for low_number in self.low_numbers:
                if low_number < len(bars): # Safety check
                    bars[low_number].set_facecolor(highlight_color)
                    bars[low_number].set_alpha(highlight_alpha)

        elif action == 43:  # Bet on High (19-36)
            for high_number in self.high_numbers:
                if high_number < len(bars): # Safety check
                    bars[high_number].set_facecolor(highlight_color)
                    bars[high_number].set_alpha(highlight_alpha)

        elif action == 44:  # Bet on First Dozen (1-12)
            for first_dozen_number in self.first_dozen_numbers:
                if first_dozen_number < len(bars): # Safety check
                    bars[first_dozen_number].set_facecolor(highlight_color)
                    bars[first_dozen_number].set_alpha(highlight_alpha)

        elif action == 45:  # Bet on Second Dozen (13-24)
            for second_dozen_number in self.second_dozen_numbers:
                if second_dozen_number < len(bars): # Safety check
                    bars[second_dozen_number].set_facecolor(highlight_color)
                    bars[second_dozen_number].set_alpha(highlight_alpha)

        elif action == 46:  # Bet on Third Dozen (25-36)
            for third_dozen_number in self.third_dozen_numbers:
                if third_dozen_number < len(bars): # Safety check
                    bars[third_dozen_number].set_facecolor(highlight_color)
                    bars[third_dozen_number].set_alpha(highlight_alpha)

        # Highlight the last extracted number
        bars[extracted_number].set_facecolor('yellow')
        bars[extracted_number].set_alpha(0.7)

        # Adjust the position of the labels to be on the outer edge
        for i, (label, angle) in enumerate(zip(labels, theta)):
            angle_label = angle + width / 2
            x = angle_label
            y = 1.05  # Position radius just outside the wheel
            angle_deg = np.degrees(angle_label)
            if angle_deg >= 270:
                angle_deg -= 360  # Normalize angle between -90 and 270 degrees
            # Text rotation
            rotation = angle_deg
            self.ax.text(x, y, str(label),
                         rotation=rotation, rotation_mode='anchor',
                         ha='center', va='center', color='black', fontsize=8,
                         clip_on=False)

        # Remove the grid and axis labels
        self.ax.grid(False)
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        # Set the radius limit to include the labels
        self.ax.set_ylim(0, 1.15)

        # Update title and texts without creating new ones
        self.title_text.set_text("Roulette Wheel - Current Spin")
        self.episode_text.set_text(f"Episode {episode+1} | Time step {time_step+1}")
        self.capital_text.set_text(f"Current capital: {self.capital} | Reward: {self.reward}")
        self.extraction_text.set_text(f"Last extracted number: {extracted_number}")

        # Draw the updated plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()        