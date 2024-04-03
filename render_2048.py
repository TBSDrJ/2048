import time
import pygame
from Game2048 import Game2048

def draw_board(
            screen: pygame.Surface, 
            game: Game2048, 
            font: pygame.font.Font,
    ) -> None:
    for i, row in enumerate(game.board):
        for j, entry in enumerate(row):
            pygame.draw.rect(screen, (20, 20, 150), 
                    (10 + j*120, 10 + i*120, 100, 100))
            if entry != 0:
                text = font.render(str(entry), True, (255, 100, 100), 
                        (20, 20, 150))
                text_loc = text.get_rect(center=(60 + j*120, 60 + i*120))
                screen.blit(text, text_loc)

def get_keys(game: Game2048, act:bool) -> bool:
    keys = pygame.key.get_pressed()
    num_keys = sum(list(keys))
    if num_keys == 0:
        act = True
    if num_keys == 1:
        if act:
            if keys[pygame.K_w]:
                game.one_turn(0)
            if keys[pygame.K_a]:
                game.one_turn(1)
            if keys[pygame.K_s]:
                game.one_turn(2)
            if keys[pygame.K_d]:
                game.one_turn(3)
        act = False
    return act

def draw_score(
            screen: pygame.Surface, 
            game: Game2048, 
            font: pygame.font.Font,
    ) -> None:
    text = font.render("SCORE: " + str(game.score), True, (255, 100, 100), 
            (20, 20, 150))
    text_loc = text.get_rect(center=(screen.get_width() / 2, 
            screen.get_height() - 25))
    pygame.draw.rect(screen, (20, 20, 150), (text_loc.left - 10, 
            text_loc.top - 10, text_loc.width + 20, text_loc.height + 20))
    screen.blit(text, text_loc)

def main():
    game = Game2048()
    pygame.init()
    screen = pygame.display.set_mode((120*game.width, 120*game.height + 50))
    font = pygame.font.Font(None, 40)
    act = True
    
    while not game.game_over:
        screen.fill((0, 200, 255))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game.game_over = True
        draw_board(screen, game, font)
        act = get_keys(game, act)
        draw_score(screen, game, font)
        pygame.display.flip()
    pygame.quit()

if __name__ == "__main__":
    main()