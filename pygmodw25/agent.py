"""
agent.py : including the main classes to create an agent. Supplementary calculations independent from class attributes
            are removed from this file.
"""
from math import atan2

import pygame
import numpy as np
from pygmodw25 import support


class AgentBase(pygame.sprite.Sprite):
    """
    Generalized agent class with very basic functionalities shared across all children
    """

    def __init__(self, id, radius, position, orientation, env_size, color, window_pad):
        """
        Initalization method of main agent class of the simulations

        :param id: ID of agent (int)
        :param radius: radius of the agent in pixels
        :param position: position of the agent bounding upper left corner in env as (x, y)
        :param orientation: absolute orientation of the agent (0 is facing to the right)
        :param env_size: environment size available for agents as (width, height)
        :param color: color of the agent as (R, G, B)
        :param window_pad: padding of the environment in simulation window in pixels
        """
        # Initializing supercalss (Pygame Sprite)
        super().__init__()
        self.id = id
        self.radius = radius
        self.position = np.array(position, dtype=np.float64)
        self.orientation = orientation
        self.orig_color = color
        self.color = self.orig_color
        self.selected_color = support.LIGHT_BLUE

        # Environment related parameters
        self.WIDTH = env_size[0]  # env width
        self.HEIGHT = env_size[1]  # env height
        self.window_pad = window_pad
        self.boundaries_x = [self.window_pad, self.window_pad + self.WIDTH]
        self.boundaries_y = [self.window_pad, self.window_pad + self.HEIGHT]

        # Boundary conditions
        # bounce_back: agents bouncing back from walls as particles
        # infinite: agents continue moving in both x and y direction and teleported to other side (torus)
        self.boundary = "infinite"

        # Non-initialisable private attributes
        self.velocity = 1  # agent absolute velocity
        self.v_max = 1  # maximum velocity of agent
        self.vx = 0  # vlocity in x direction
        self.vy = 0  # velocity in y direction

        # Time step
        self.dt = 0.05
        self.dv = 0
        self.dtheta = 0

        # Interaction
        self.is_moved_with_cursor = 0

        # Visualization parameters
        self.show_stats = False
        self.change_color_with_orientation = False

        # Initial Visualization of agent
        self.image = pygame.Surface([radius * 2, radius * 2])
        self.image.fill(support.BACKGROUND)
        self.image.set_colorkey(support.BACKGROUND)
        pygame.draw.circle(
            self.image, color, (radius, radius), radius
        )

        # Showing agent orientation with a line towards agent orientation
        pygame.draw.line(self.image, support.BACKGROUND, (radius, radius),
                         ((1 + np.cos(self.orientation)) * radius, (1 - np.sin(self.orientation)) * radius), 3)
        self.rect = self.image.get_rect()
        self.rect.x = self.position[0]
        self.rect.y = self.position[1]
        self.mask = pygame.mask.from_surface(self.image)

    def change_color(self):
        """Changing color of agent according to the behavioral mode the agent is currently in."""
        self.color = support.calculate_color(self.orientation, self.velocity)

    def move_with_mouse(self, mouse, left_state, right_state):
        """Moving the agent with the mouse cursor, and rotating"""
        if self.rect.collidepoint(mouse):
            # setting position of agent to cursor position
            self.position[0] = mouse[0] - self.radius
            self.position[1] = mouse[1] - self.radius
            if left_state:
                self.orientation += 0.1
            if right_state:
                self.orientation -= 0.1
            self.prove_orientation()
            self.is_moved_with_cursor = 1
            # updating agent visualization to make it more responsive
            self.draw_update()
        else:
            self.is_moved_with_cursor = 0

    def draw_update(self):
        """
        updating the outlook of the agent according to position and orientation
        """
        # update position
        self.rect.x = self.position[0]
        self.rect.y = self.position[1]

        # change agent color according to mode
        if self.change_color_with_orientation:
            self.change_color()

        # update surface according to new orientation
        # creating visualization surface for agent as a filled circle
        self.image = pygame.Surface([self.radius * 2, self.radius * 2])
        self.image.fill(support.BACKGROUND)
        self.image.set_colorkey(support.BACKGROUND)
        try:
            if self.is_moved_with_cursor:
                pygame.draw.circle(
                    self.image, self.selected_color, (self.radius, self.radius), self.radius
                )
            else:
                pygame.draw.circle(
                    self.image, self.color, (self.radius, self.radius), self.radius
                )
        except:
            self.color[3] = 0
            pygame.draw.circle(
                self.image, self.color, (self.radius, self.radius), self.radius
            )

        # showing agent orientation with a line towards agent orientation
        pygame.draw.line(self.image, support.BACKGROUND, (self.radius, self.radius),
                         ((1 + np.cos(self.orientation)) * self.radius, (1 - np.sin(self.orientation)) * self.radius),
                         3)
        self.mask = pygame.mask.from_surface(self.image)

    def reflect_from_walls(self, boundary_condition):
        """reflecting agent from environment boundaries according to a desired x, y coordinate. If this is over any
        boundaries of the environment, the agents position and orientation will be changed such that the agent is
         reflected from these boundaries."""

        # Boundary conditions according to center of agent (simple)
        x = self.position[0] + self.radius
        y = self.position[1] + self.radius

        if boundary_condition == "bounce_back":
            # Reflection from left wall
            if x < self.boundaries_x[0]:
                self.position[0] = self.boundaries_x[0] - self.radius

                if np.pi / 2 <= self.orientation < np.pi:
                    self.orientation -= np.pi / 2
                elif np.pi <= self.orientation <= 3 * np.pi / 2:
                    self.orientation += np.pi / 2
                self.prove_orientation()  # bounding orientation into 0 and 2pi

            # Reflection from right wall
            if x > self.boundaries_x[1]:

                self.position[0] = self.boundaries_x[1] - self.radius - 1

                if 3 * np.pi / 2 <= self.orientation < 2 * np.pi:
                    self.orientation -= np.pi / 2
                elif 0 <= self.orientation <= np.pi / 2:
                    self.orientation += np.pi / 2
                self.prove_orientation()  # bounding orientation into 0 and 2pi

            # Reflection from upper wall
            if y < self.boundaries_y[0]:
                self.position[1] = self.boundaries_y[0] - self.radius

                if np.pi / 2 <= self.orientation <= np.pi:
                    self.orientation += np.pi / 2
                elif 0 <= self.orientation < np.pi / 2:
                    self.orientation -= np.pi / 2
                self.prove_orientation()  # bounding orientation into 0 and 2pi

            # Reflection from lower wall
            if y > self.boundaries_y[1]:
                self.position[1] = self.boundaries_y[1] - self.radius - 1
                if 3 * np.pi / 2 <= self.orientation <= 2 * np.pi:
                    self.orientation += np.pi / 2
                elif np.pi <= self.orientation < 3 * np.pi / 2:
                    self.orientation -= np.pi / 2
                self.prove_orientation()  # bounding orientation into 0 and 2pi

        elif boundary_condition == "infinite":

            if x < self.boundaries_x[0]:
                self.position[0] = self.boundaries_x[1] - self.radius
            elif x > self.boundaries_x[1]:
                self.position[0] = self.boundaries_x[0] + self.radius

            if y < self.boundaries_y[0]:
                self.position[1] = self.boundaries_y[1] - self.radius
            elif y > self.boundaries_y[1]:
                self.position[1] = self.boundaries_y[0] + self.radius

    def prove_orientation(self):
        """Restricting orientation angle between 0 and 2 pi"""
        if self.orientation < 0:
            self.orientation = 2 * np.pi + self.orientation
        if self.orientation > np.pi * 2:
            self.orientation = self.orientation - 2 * np.pi

    def prove_velocity(self):
        """Restricting the absolute velocity of the agent"""
        vel_sign = np.sign(self.velocity)
        if vel_sign == 0:
            vel_sign = +1
        if np.abs(self.velocity) > self.v_max:
            # stopping agent if too fast during exploration
            self.velocity = self.v_max

    def update(self, agents):
        """
        main update method of the agent. This method is called in every timestep to calculate the new state/position
        of the agent and visualize it in the environment
        :param agents: a list of all other agents in the environment.
        """
        if not self.is_moved_with_cursor:  # we freeze agents when we move them
            # # updating agent's state variables according to calculated vel and theta
            self.orientation += self.dt * self.dtheta
            self.prove_orientation()  # bounding orientation into 0 and 2pi
            self.velocity += self.dt * self.dv
            self.prove_velocity()  # possibly bounding velocity of agent

            # updating agent's position
            self.vx = self.velocity * np.cos(self.orientation)
            self.vy = self.velocity * np.sin(self.orientation)
            self.position[0] += self.vx
            self.position[1] -= self.vy

            # boundary conditions if applicable
            self.reflect_from_walls(self.boundary)

        # updating agent visualization
        self.draw_update()

