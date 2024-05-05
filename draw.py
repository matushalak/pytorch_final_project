import pygame
import sys
from numpy.random import randint
import os

## todo: improve smooth linking by lines

# Initialize Pygame
pygame.init()

# Set the dimensions of each square and the grid size
square_size = 10
# 2x mnist resolution
grid_size = (56, 56)
window_size = (square_size * grid_size[0], square_size * grid_size[1])

# Set colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)

# Create the display window
screen = pygame.display.set_mode(window_size)
pygame.display.set_caption("Squared Paper Drawing")

# Create a 2D list to store the state of each square
grid_state = [[False for _ in range(grid_size[1])] for _ in range(grid_size[0])]


# Font for text input
font = pygame.font.Font(None, 60)
text_color = BLACK

# Variables for label input field
label = ''
input_rect = pygame.Rect(10, window_size[1] - 80, 400, 40)
active = True

# Function to draw the grid
def draw_grid():
    for y in range(grid_size[1]):
        for x in range(grid_size[0]):
            rect = pygame.Rect(x * square_size, y * square_size, square_size, square_size)
            if grid_state[x][y]:
                pygame.draw.rect(screen, BLACK, rect)
            else:
                pygame.draw.rect(screen, WHITE, rect)
            pygame.draw.rect(screen, GRAY, rect, 1)  # Draw grid lines

# Function to reset the canvas
def reset_canvas():
    for x in range(grid_size[0]):
        for y in range(grid_size[1]):
            grid_state[x][y] = False

# Function to save the drawing as a PNG file
def save_drawing(label):
    # Count how many files with the same label are already saved in the directory
    count = 0
    for filename in os.listdir('drawing'):
        if filename.startswith(label):
            count += 1
    # Create filename
    filename = f"{label}_{count}.png"
    
    # Create a surface to draw the saved image
    saved_surface = pygame.Surface(window_size)
    saved_surface.fill(WHITE)
    for y in range(grid_size[1]):
        for x in range(grid_size[0]):
            if grid_state[x][y]:
                pygame.draw.rect(saved_surface, BLACK, (x * square_size, y * square_size, square_size, square_size))
    # Save the surface to a PNG file
    ## todo: change name to be based on number of images already there    
    pygame.image.save(saved_surface, f'drawing/{filename}')
    # Reset the canvas
    reset_canvas()

# Function to fill the center square and its immediate neighbors
def fill_neighbors(x, y):
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            nx = x + dx
            ny = y + dy
            if 0 <= nx < grid_size[0] and 0 <= ny < grid_size[1]:
                grid_state[nx][ny] = True

# Define reset and save button rectangles
reset_button_rect = pygame.Rect(10, 10, 100, 30)
save_button_rect = pygame.Rect(120, 10, 100, 30)

# Main loop
running = True
dragging = False
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_x, mouse_y = event.pos
            # Convert mouse position to grid indices
            grid_x = mouse_x // square_size
            grid_y = mouse_y // square_size
            if reset_button_rect.collidepoint(event.pos):
                reset_canvas()
                label = ''
            elif save_button_rect.collidepoint(event.pos):
                save_drawing(label)
                label = ''
            elif input_rect.collidepoint(event.pos):
                active = True
            elif 0 <= grid_x < grid_size[0] and 0 <= grid_y < grid_size[1]:
                grid_state[grid_x][grid_y] = True
                fill_neighbors(grid_x, grid_y)
                dragging = True
            else:
                active = False
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            dragging = False
        elif event.type == pygame.MOUSEMOTION and dragging:
            mouse_x, mouse_y = event.pos
            # Convert mouse position to grid indices
            grid_x = mouse_x // square_size
            grid_y = mouse_y // square_size
            if 0 <= grid_x < grid_size[0] and 0 <= grid_y < grid_size[1]:
                grid_state[grid_x][grid_y] = True
                fill_neighbors(grid_x, grid_y)
        elif event.type == pygame.KEYDOWN:
            if active:
                if event.key == pygame.K_RETURN:
                    save_drawing(label)
                    label = ''
                elif event.key == pygame.K_BACKSPACE:
                    label = label[:-1]
                else:
                    label += event.unicode

    # Fill the background with white
    screen.fill(WHITE)
    # Draw the grid
    draw_grid()
    
    # Draw reset button
    pygame.draw.rect(screen, GRAY, reset_button_rect)
    font = pygame.font.SysFont(None, 24)
    text = font.render("Reset", True, BLACK)
    text_rect = text.get_rect(center=reset_button_rect.center)
    screen.blit(text, text_rect)
    
    # Draw save button
    pygame.draw.rect(screen, GRAY, save_button_rect)
    text = font.render("Save", True, BLACK)
    text_rect = text.get_rect(center=save_button_rect.center)
    screen.blit(text, text_rect)
    
    # Define the message text
    message = "Please type in a label for the drawing before saving"
    
    # Render the message text
    message_surface = font.render(message, True, BLACK)
    
    # Calculate the position to center the message above the input field
    message_x = input_rect.x + input_rect.width // 2 - message_surface.get_width() // 2
    message_y = input_rect.y - 30  # Place the message above the input field, adjust as needed
    
    # Draw the message text
    screen.blit(message_surface, (message_x, message_y))
    
    # Draw the label input field
    pygame.draw.rect(screen, WHITE, input_rect)  # Fill input field with white
    pygame.draw.rect(screen, BLACK, input_rect, 2)
    text_surface = font.render(label, True, text_color)
    screen.blit(text_surface, (input_rect.x + 5, input_rect.y + 5))
    
    # Update the display
    pygame.display.flip()

# Quit Pygame
pygame.quit()
sys.exit()
