#include <eepp/ee.hpp>

EE::Window::Window* win = NULL;

void mainLoop() {
	// Clear the screen buffer
	win->clear();

	// Create an instance of the primitive renderer
	Primitives p;

	// Change the color
	p.setColor( Color( 0, 255, 0, 150 ) );

	// Update the input
	win->getInput()->update();

	// Check if ESCAPE key is pressed
	if ( win->getInput()->isKeyDown( KEY_ESCAPE ) ) {
		// Close the window
		win->close();
	}

	// Draw a circle
	p.drawCircle( win->getCenter(), 200, 60 );

	// Draw frame
	win->display();
}

// EE_MAIN_FUNC is needed by some platforms to be able to find the real application main
EE_MAIN_FUNC int main( int, char*[] ) {
	// Create a new window with vsync enabled
	win = Engine::instance()->createWindow( WindowSettings( 960, 640, "eepp - Empty Window" ),
											ContextSettings( true ) );

	// Check if created
	if ( win->isOpen() ) {
		// Set window background color
		win->setClearColor( RGB( 50, 50, 50 ) );

		// Set the MainLoop function and run it
		// This is the application loop, it will loop until the window is closed.
		// This is only a requirement if you want to support Emscripten builds ( WebGL + Canvas ).
		// This is the same as, except for Emscripten.
		// while ( win->isRunning() )
		// {
		//		MainLoop();
		// }
		win->runMainLoop( &mainLoop );
	}

	// Destroy the engine instance. Destroys all the windows and engine singletons.
	Engine::destroySingleton();

	// If was compiled in debug mode it will print the memory manager report
	MemoryManager::showResults();

	return EXIT_SUCCESS;
}
