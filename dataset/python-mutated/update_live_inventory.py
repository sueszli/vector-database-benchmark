import logging
from datetime import datetime, timedelta
from pokemongo_bot import inventory
from pokemongo_bot.base_task import BaseTask
from pokemongo_bot.worker_result import WorkerResult
from pokemongo_bot.tree_config_builder import ConfigException

class UpdateLiveInventory(BaseTask):
    """
    Periodically displays the user inventory in the terminal.

    Example config :
    {
        "type": "UpdateLiveInventory",
        "config": {
          "enabled": true,
          "min_interval": 120,
          "show_all_multiple_lines": false,
          "items": ["space_info", "pokeballs", "greatballs", "ultraballs", "razzberries", "luckyegg"]
        }
    }

    min_interval : The minimum interval at which the stats are displayed,
                   in seconds (defaults to 120 seconds).
                   The update interval cannot be accurate as workers run synchronously.
    show_all_multiple_lines : Logs all items on inventory using multiple lines.
                              Ignores configuration of 'items' 
    items : An array of items to display and their display order (implicitly),
            see available items below (defaults to []).

    Available items :
        'pokemon_bag': pokemon in inventory (i.e. 'Pokemon Bag: 100/250')
        'space_info': shows inventory bag space (i.e. 'Items: 140/350')
        'pokeballs'
        'greatballs'
        'ultraballs'
        'masterballs'
        'razzberries'
        'nanabberries'
        'pinapberries'
        'goldenrazzberries'
        'goldennanabberries'
        'goldenpinapberries'
        'luckyegg'
        'incubator'
        'incubatorsuper'
        'troydisk'
        'potion'
        'superpotion'
        'hyperpotion'
        'maxpotion'
        'incense'
        'revive'
        'maxrevive'
        'sunstone'
        'kingsrock'
        'metalcoat'
        'dragonscale'
        'upgrade'
        'starpiece'
    """
    SUPPORTED_TASK_API_VERSION = 1

    def initialize(self):
        if False:
            i = 10
            return i + 15
        self.next_update = None
        self.min_interval = self.config.get('min_interval', 120)
        self.show_all_multiple_lines = self.config.get('show_all_multiple_lines', False)
        self.displayed_items = self.config.get('items', [])
        self.logger = logging.getLogger(type(self).__name__)
        if self.show_all_multiple_lines:
            self.bot.event_manager.register_event('show_inventory')
        else:
            self.bot.event_manager.register_event('show_inventory', parameters=('items',))

    def work(self):
        if False:
            return 10
        '\n        Displays the items if necessary.\n        :return: Always returns WorkerResult.SUCCESS.\n        :rtype: WorkerResult\n        '
        if not self.should_print():
            return WorkerResult.SUCCESS
        self.inventory = inventory.items()
        if self.show_all_multiple_lines:
            self.print_all()
            self.print_inv(self.get_inventory_line(True), True)
            return WorkerResult.SUCCESS
        line = self.get_inventory_line()
        if not line:
            return WorkerResult.SUCCESS
        self.print_inv(line)
        return WorkerResult.SUCCESS

    def should_print(self):
        if False:
            return 10
        '\n        Returns a value indicating whether the items should be displayed.\n        :return: True if the stats should be displayed; otherwise, False.\n        :rtype: bool\n        '
        return self.next_update is None or datetime.now() >= self.next_update

    def compute_next_update(self):
        if False:
            print('Hello World!')
        '\n        Computes the next update datetime based on the minimum update interval.\n        :return: Nothing.\n        :rtype: None\n        '
        self.next_update = datetime.now() + timedelta(seconds=self.min_interval)

    def print_inv(self, items, is_debug=False):
        if False:
            return 10
        '\n        Logs the items into the terminal using an event.\n        :param items: The items to display.\n        :type items: string\n        :param is_debug: If True emits event at debug level.\n        :type is_debug: boolean\n        :return: Nothing.\n        :rtype: None\n        '
        if not is_debug:
            self.emit_event('show_inventory', formatted='{items}', data={'items': items})
        else:
            self.emit_event('show_inventory', sender=self, level='debug', formatted='{items}', data={'items': items})
        self.compute_next_update()

    def get_inventory_line(self, is_debug=False):
        if False:
            i = 10
            return i + 15
        '\n        Generates a items string according to the configuration.\n        :param is_debug: If True returns a string with all items.\n        :type is_debug: boolean\n        :return: A string containing items and their count, ready to be displayed.\n        :rtype: string\n        '
        available_items = {'pokemon_bag': 'Pokemon: {:,}/{:,}'.format(inventory.Pokemons.get_space_used(), inventory.get_pokemon_inventory_size()), 'space_info': 'Items: {:,}/{:,}'.format(self.inventory.get_space_used(), self.inventory.get_space_used() + self.inventory.get_space_left()), 'pokeballs': 'Pokeballs: {:,}'.format(self.inventory.get(1).count), 'greatballs': 'Greatballs: {:,}'.format(self.inventory.get(2).count), 'ultraballs': 'Ultraballs: {:,}'.format(self.inventory.get(3).count), 'masterballs': 'Masterballs: {:,}'.format(self.inventory.get(4).count), 'razzberries': 'Razz Berries: {:,}'.format(self.inventory.get(701).count), 'nanabberries': 'Nanab Berries: {:,}'.format(self.inventory.get(703).count), 'pinapberries': 'Pinap Berries: {:,}'.format(self.inventory.get(705).count), 'goldenrazzberries': 'Golden Razz Berries: {:,}'.format(self.inventory.get(706).count), 'goldennanabberries': 'Golden Nanab Berries: {:,}'.format(self.inventory.get(707).count), 'goldenpinapberries': 'Golden Pinap Berries: {:,}'.format(self.inventory.get(708).count), 'luckyegg': 'Lucky Egg: {:,}'.format(self.inventory.get(301).count), 'incubator': 'Incubator: {:,}'.format(self.inventory.get(902).count), 'incubatorsuper': 'Super Incubator: {:,}'.format(self.inventory.get(903).count), 'troydisk': 'Troy Disk: {:,}'.format(self.inventory.get(501).count), 'potion': 'Potion: {:,}'.format(self.inventory.get(101).count), 'superpotion': 'Super Potion: {:,}'.format(self.inventory.get(102).count), 'hyperpotion': 'Hyper Potion: {:,}'.format(self.inventory.get(103).count), 'maxpotion': 'Max Potion: {:,}'.format(self.inventory.get(104).count), 'incense': 'Incense: {:,}'.format(self.inventory.get(401).count), 'revive': 'Revive: {:,}'.format(self.inventory.get(201).count), 'maxrevive': 'Max Revive: {:,}'.format(self.inventory.get(202).count), 'sunstone': 'Sun Stone: {:,}'.format(self.inventory.get(1101).count), 'kingsrock': 'Kings Rock: {:,}'.format(self.inventory.get(1102).count), 'metalcoat': 'Metal Coat: {:,}'.format(self.inventory.get(1103).count), 'dragonscale': 'Dragon Scale: {:,}'.format(self.inventory.get(1104).count), 'upgrade': 'Upgrade: {:,}'.format(self.inventory.get(1105).count), 'starpiece': 'Star Piece: {:,}'.format(self.inventory.get(1404).count)}

        def get_item(item):
            if False:
                return 10
            "\n            Fetches a item string from the available items dictionary.\n            :param item: The item name.\n            :type item: string\n            :return: The generated item string.\n            :rtype: string\n            :raise: ConfigException: When the provided item string isn't in the available items\n            dictionary.\n            "
            if item not in available_items:
                raise ConfigException("item '{}' isn't available for displaying".format(item))
            return available_items[item]
        if is_debug:
            temp = []
            for (key, value) in available_items.iteritems():
                temp.append(value)
            return ' | '.join(temp)
        line = ' | '.join(map(get_item, self.displayed_items))
        return line

    def print_all(self):
        if False:
            i = 10
            return i + 15
        '\n        Logs the items into the terminal using self.logger.\n        It logs using multiple lines and logs all items.\n        :return: Nothing.\n        :rtype: None\n        '
        self.logger.info('Pokemon Bag: {}/{}'.format(inventory.Pokemons.get_space_used(), inventory.get_pokemon_inventory_size()))
        self.logger.info('Items: {}/{}'.format(self.inventory.get_space_used(), inventory.get_item_inventory_size()))
        self.logger.info('Poke Balls: {} | Great Balls: {} | Ultra Balls: {} | Master Balls: {}'.format(self.inventory.get(1).count, self.inventory.get(2).count, self.inventory.get(3).count, self.inventory.get(4).count))
        self.logger.info('Razz Berries: {} | Nanab Berries: {} | Pinap Berries: {} | Golden Razz Berries: {} | Golden Nanab Berries: {} | Pinap Berries: {}'.format(self.inventory.get(701).count, self.inventory.get(703).count, self.inventory.get(705).count, self.inventory.get(706).count, self.inventory.get(707).count, self.inventory.get(708).count))
        self.logger.info('Incubator: {} | Super Incubator: {}'.format(self.inventory.get(902).count, self.inventory.get(903).count))
        self.logger.info('Potion: {} | Super Potion: {} | Hyper Potion: {} | Max Potion: {}'.format(self.inventory.get(101).count, self.inventory.get(102).count, self.inventory.get(103).count, self.inventory.get(104).count))
        self.logger.info('Lucky Egg: {} | Incense: {} | Troy Disk: {} | Star Pieces: {}'.format(self.inventory.get(301).count, self.inventory.get(401).count, self.inventory.get(501).count, self.inventory.get(1404).count))
        self.logger.info('Revive: {} | Max Revive: {}'.format(self.inventory.get(201).count, self.inventory.get(202).count))
        self.logger.info('Sun Stone: {} | Kings Rock: {} | Metal Coat: {} | Dragon Scale: {} | Upgrade: {}'.format(self.inventory.get(1101).count, self.inventory.get(1102).count, self.inventory.get(1103).count, self.inventory.get(1104).count, self.inventory.get(1105).count))
        self.compute_next_update()