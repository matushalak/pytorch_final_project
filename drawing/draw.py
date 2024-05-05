import pygame
import sys
from numpy.random import randint

## todo: improve smooth linking by lines

# Initialize Pygame
pygame.init()

# Set the dimensions of each square and the grid size
square_size = 10
grid_size = (60, 60)
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
def save_drawing():
    # Create a surface to draw the saved image
    saved_surface = pygame.Surface(window_size)
    saved_surface.fill(WHITE)
    for y in range(grid_size[1]):
        for x in range(grid_size[0]):
            if grid_state[x][y]:
                pygame.draw.rect(saved_surface, BLACK, (x * square_size, y * square_size, square_size, square_size))
    # Save the surface to a PNG file
    ## todo: change name to be based on number of images already there    
    pygame.image.save(saved_surface, f"drawing_{randint(0,10000)}.png")
    # Reset the canvas
    reset_canvas()

# Main loop
running = True
dragging = False
reset_button_rect = pygame.Rect(10, 10, 100, 30)
save_button_rect = pygame.Rect(120, 10, 100, 30)
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
            elif save_button_rect.collidepoint(event.pos):
                save_drawing()
            elif 0 <= grid_x < grid_size[0] and 0 <= grid_y < grid_size[1]:
                grid_state[grid_x][grid_y] = True
                dragging = True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            dragging = False
        elif event.type == pygame.MOUSEMOTION and dragging:
            mouse_x, mouse_y = event.pos
            # Convert mouse position to grid indices
            grid_x = mouse_x // square_size
            grid_y = mouse_y // square_size
            if 0 <= grid_x < grid_size[0] and 0 <= grid_y < grid_size[1]:
                grid_state[grid_x][grid_y] = True

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
    # Update the display
    pygame.display.flip()

# Quit Pygame
pygame.quit()
sys.exit()
