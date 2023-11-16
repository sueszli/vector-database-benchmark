"""
Cocoa window cannot be destroyed programmatically, until it finishes processing a NSEvent
So we need to simulate a mouse movement in order to generate an event.
"""
from Quartz.CoreGraphics import CGEventCreate, CGEventCreateMouseEvent, CGEventGetLocation, CGEventPost, kCGEventLeftMouseDown, kCGEventLeftMouseUp, kCGEventMouseMoved, kCGHIDEventTap, kCGMouseButtonLeft

def mousePos():
    if False:
        return 10
    event = CGEventCreate(None)
    pointer = CGEventGetLocation(event)
    return (pointer.x, pointer.y)

def mouseEvent(type, posx, posy):
    if False:
        return 10
    theEvent = CGEventCreateMouseEvent(None, type, (posx, posy), kCGMouseButtonLeft)
    CGEventPost(kCGHIDEventTap, theEvent)

def mouseMove(posx, posy):
    if False:
        for i in range(10):
            print('nop')
    mousePos()
    mouseEvent(kCGEventMouseMoved, posx, posy)

def mouseMoveRelative(dx, dy):
    if False:
        return 10
    (posx, posy) = mousePos()
    mouseMove(posx + dx, posy + dy)

def mouseclick(posx, posy):
    if False:
        return 10
    mouseEvent(kCGEventLeftMouseDown, posx, posy)
    mouseEvent(kCGEventLeftMouseUp, posx, posy)
if __name__ == '__main__':
    mouseMoveRelative(100, 100)