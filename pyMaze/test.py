# from pygame.locals import *
# import pygame
 
# class Player:
#     x = 10
#     y = 10
#     speed = 0.02
 
#     def moveRight(self):
#         self.x = self.x + self.speed
 
#     def moveLeft(self):
#         self.x = self.x - self.speed
 
#     def moveUp(self):
#         self.y = self.y - self.speed
 
#     def moveDown(self):
#         self.y = self.y + self.speed
 
# class Maze:
#     def __init__(self):
#        self.M = 10
#        self.N = 8
#        self.maze = [ 1,1,1,1,1,1,1,1,1,1,
#                      1,0,0,0,0,0,0,0,0,1,
#                      1,0,0,0,0,0,0,0,0,1,
#                      1,0,1,1,1,1,1,1,0,1,
#                      1,0,1,0,0,0,0,0,0,1,
#                      1,0,1,0,1,1,1,1,0,1,
#                      1,0,0,0,0,0,0,0,0,1,
#                      1,1,1,1,1,1,1,1,1,1,]
 
#     def draw(self,display_surf,image_surf):
#        bx = 0
#        by = 0
#        for i in range(0,self.M*self.N):
#            if self.maze[ bx + (by*self.M) ] == 1:
#                display_surf.blit(image_surf,( bx * 10 , by * 10))
 
#            bx = bx + 1
#            if bx > self.M-1:
#                bx = 0 
#                by = by + 1
 
 
# class App:
 
#     windowWidth = 400
#     windowHeight = 300
#     player = 0
 
#     def __init__(self):
#         self._running = True
#         self._display_surf = None
#         self._image_surf = None
#         self._block_surf = None
#         self.player = Player()
#         self.maze = Maze()
 
#     def on_init(self):
#         pygame.init()
#         self._display_surf = pygame.display.set_mode((self.windowWidth,self.windowHeight), pygame.HWSURFACE)
 
#         pygame.display.set_caption('Pygame pythonspot.com example')
#         self._running = True
#         self._image_surf = pygame.image.load("player.png").convert()
#         self._block_surf = pygame.image.load("block.png").convert()
 
#     def on_event(self, event):
#         if event.type == QUIT:
#             self._running = False
 
#     def on_loop(self):
#         pass
 
#     def on_render(self):
#         self._display_surf.fill((0,0,0))
#         self._display_surf.blit(self._image_surf,(self.player.x,self.player.y))
#         self.maze.draw(self._display_surf, self._block_surf)
#         pygame.display.flip()
 
#     def on_cleanup(self):
#         pygame.quit()
 
#     def on_execute(self):
#         if self.on_init() == False:
#             self._running = False
 
#         while( self._running ):
#             pygame.event.pump()
#             keys = pygame.key.get_pressed()
 
#             if (keys[K_RIGHT]):
#                 self.player.moveRight()
 
#             if (keys[K_LEFT]):
#                 self.player.moveLeft()
 
#             if (keys[K_UP]):
#                 self.player.moveUp()
 
#             if (keys[K_DOWN]):
#                 self.player.moveDown()
 
#             if (keys[K_ESCAPE]):
#                 self._running = False
 
#             self.on_loop()
#             self.on_render()
#         self.on_cleanup()
 
# if __name__ == "__main__" :
#     theApp = App()
#     theApp.on_execute()





import pygame

pygame.init()
screen = pygame.display.set_mode((300, 300))
ck = (127, 33, 33)
size = 25
while True:
  if pygame.event.get(pygame.MOUSEBUTTONDOWN):
    s = pygame.Surface((50, 50))

    # first, "erase" the surface by filling it with a color and
    # setting this color as colorkey, so the surface is empty
    s.fill(ck)
    s.set_colorkey(ck)

    pygame.draw.circle(s, (255, 0, 0), (size, size), size, 2)

    # after drawing the circle, we can set the 
    # alpha value (transparency) of the surface
    s.set_alpha(75)

    x, y = pygame.mouse.get_pos()
    screen.blit(s, (x-size, y-size))

  pygame.event.poll()
  pygame.display.flip()