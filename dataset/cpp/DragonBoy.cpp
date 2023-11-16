/** @file DragonBoy.cpp
 ** @author Piotr Krupa (piotrkrupa06@gmail.com)
 ** @license MIT License
 **/

#include <SFML/Graphics.hpp>

#include <dragonBones/SFMLFactory.h>
#include <dragonBones/SFMLArmatureDisplay.h>


int main()
{
	sf::RenderWindow window(sf::VideoMode(1024, 768), "My window");
	window.setFramerateLimit(60);

	dragonBones::SFMLFactory factory;

	sf::Texture texture;
	texture.loadFromFile("DragonBoy_tex.png");

	factory.loadDragonBonesData("DragonBoy_ske.json");
	factory.loadTextureAtlasData("DragonBoy_tex.json", &texture);

	auto armatureDisplay = new dragonBones::SFMLArmatureDisplay("Dragon");
	armatureDisplay->getAnimation()->play("walk");
	armatureDisplay->setPosition({ 512.f, 440.f });

	sf::Clock clock;

	while (window.isOpen())
	{
		float deltaTime = clock.restart().asSeconds();

		sf::Event event;
		while (window.pollEvent(event))
		{
			if (event.type == sf::Event::Closed)
				window.close();
		}

		factory.update(deltaTime);

		window.clear();
		window.draw(*armatureDisplay);
		window.display();
	}
	
	return 0;
}
