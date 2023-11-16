from __future__ import unicode_literals
import os
import json
from pokemongo_bot.base_task import BaseTask
from pokemongo_bot.human_behaviour import sleep, action_delay
from pokemongo_bot.inventory import Attack
from pokemongo_bot.inventory import Pokemon
from pokemongo_bot.inventory import pokemons
import re
DEFAULT_IGNORE_FAVORITES = False
DEFAULT_GOOD_ATTACK_THRESHOLD = 0.7
DEFAULT_TEMPLATE = '{name}'
DEFAULT_NICKNAME_WAIT_MIN = 3
DEFAULT_NICKNAME_WAIT_MAX = 3
MAXIMUM_NICKNAME_LENGTH = 12

class NicknamePokemon(BaseTask):
    SUPPORTED_TASK_API_VERSION = 1
    '\n    Nickname user pokemons according to the specified template\n\n\n    PARAMETERS:\n\n    dont_nickname_favorite (default: False)\n        Prevents renaming of favorited pokemons\n\n    good_attack_threshold (default: 0.7)\n        Threshold for perfection of the attack in it\'s type (0.0-1.0)\n         after which attack will be treated as good.\n        Used for {fast_attack_char}, {charged_attack_char}, {attack_code}\n         templates\n\n    nickname_template (default: \'{name}\')\n        Template for nickname generation.\n        Empty template or any resulting in the simple pokemon name\n         (e.g. \'\', \'{name}\', ...) will revert all pokemon to their original\n         names (as if they had no nickname).\n\n        Niantic imposes a 12-character limit on all pokemon nicknames, so\n         any new nickname will be truncated to 12 characters if over that limit.\n        Thus, it is up to the user to exercise judgment on what template will\n         best suit their need with this constraint in mind.\n\n        You can use full force of the Python [Format String syntax](https://docs.python.org/2.7/library/string.html#formatstrings)\n        For example, using `{name:.8s}` causes the Pokemon name to never take up\n         more than 8 characters in the nickname. This would help guarantee that\n         a template like `{name:.8s}_{iv_pct}` never goes over the 12-character\n         limit.\n\n\n    **NOTE:** If you experience frequent `Pokemon not found` error messages,\n     this is because the inventory cache has not been updated after a pokemon\n     was released. This can be remedied by placing the `NicknamePokemon` task\n     above the `TransferPokemon` task in your `config.json` file.\n\n\n    EXAMPLE CONFIG:\n    {\n      "type": "NicknamePokemon",\n      "config": {\n        "enabled": true,\n        "dont_nickname_favorite": false,\n        "good_attack_threshold": 0.7,\n        "nickname_template": "{iv_pct}-{iv_ads}"\n      }\n    }\n\n\n    SUPPORTED PATTERN KEYS:\n\n    {name}  Pokemon name      (e.g. Articuno)\n    {id}    Pokemon ID/Number (1-151)\n    {cp}    Combat Points     (10-4145)\n\n    # Individial Values\n    {iv_attack}  Individial Attack (0-15) of the current specific pokemon\n    {iv_defense} Individial Defense (0-15) of the current specific pokemon\n    {iv_stamina} Individial Stamina (0-15) of the current specific pokemon\n    {iv_ads}     Joined IV values (e.g. 4/12/9)\n    {iv_sum}     Sum of the Individial Values (0-45)\n    {iv_pct}     IV perfection (in 000-100 format - 3 chars)\n    {iv_pct2}    IV perfection (in 00-99 format - 2 chars)\n                    So 99 is best (it\'s a 100% perfection)\n    {iv_pct1}    IV perfection (in 0-9 format - 1 char)\n    {iv_ads_hex} Joined IV values in HEX (e.g. 4C9)\n\n    # Basic Values of the pokemon (identical for all of one kind)\n    {base_attack}   Basic Attack (40-284) of the current pokemon kind\n    {base_defense}  Basic Defense (54-242) of the current pokemon kind\n    {base_stamina}  Basic Stamina (20-500) of the current pokemon kind\n    {base_ads}      Joined Basic Values (e.g. 125/93/314)\n\n    # Final Values of the pokemon (Base Values + Individial Values)\n    {attack}        Basic Attack + Individial Attack\n    {defense}       Basic Defense + Individial Defense\n    {stamina}       Basic Stamina + Individial Stamina\n    {sum_ads}       Joined Final Values (e.g. 129/97/321)\n\n    # IV CP perfection - it\'s a kind of IV perfection percent\n    #  but calculated using weight of each IV in its contribution\n    #  to CP of the best evolution of current pokemon.\n    # So it tends to be more accurate than simple IV perfection.\n    {ivcp_pct}      IV CP perfection (in 000-100 format - 3 chars)\n    {ivcp_pct2}     IV CP perfection (in 00-99 format - 2 chars)\n                        So 99 is best (it\'s a 100% perfection)\n    {ivcp_pct1}     IV CP perfection (in 0-9 format - 1 char)\n\n    # Character codes for fast/charged attack types.\n    # If attack is good character is uppecased, otherwise lowercased.\n    # Use \'good_attack_threshold\' option for customization\n    #\n    # It\'s an effective way to represent type with one character.\n    #   If first char of the type name is unique - use it,\n    #    in other case suitable substitute used\n    #\n    # Type codes:\n    #   Bug: \'B\'\n    #   Dark: \'K\'\n    #   Dragon: \'D\'\n    #   Electric: \'E\'\n    #   Fairy: \'Y\'\n    #   Fighting: \'T\'\n    #   Fire: \'F\'\n    #   Flying: \'L\'\n    #   Ghost: \'H\'\n    #   Grass: \'A\'\n    #   Ground: \'G\'\n    #   Ice: \'I\'\n    #   Normal: \'N\'\n    #   Poison: \'P\'\n    #   Psychic: \'C\'\n    #   Rock: \'R\'\n    #   Steel: \'S\'\n    #   Water: \'W\'\n    #\n    {fast_attack_char}      One character code for fast attack type\n                                (e.g. \'F\' for good Fire or \'s\' for bad\n                                Steel attack)\n    {charged_attack_char}   One character code for charged attack type\n                                (e.g. \'n\' for bad Normal or \'I\' for good\n                                Ice attack)\n    {attack_code}           Joined 2 character code for both attacks\n                                (e.g. \'Lh\' for pokemon with good Flying\n                                and weak Ghost attacks)\n\n    # Moveset perfection percents for attack and for defense\n    #  Calculated for current pokemon only, not between all pokemons\n    #  So perfect moveset can be weak if pokemon is weak (e.g. Caterpie)\n    {attack_pct}   Moveset perfection for attack (in 000-100 format - 3 chars)\n    {defense_pct}  Moveset perfection for defense (in 000-100 format - 3 chars)\n    {attack_pct2}  Moveset perfection for attack (in 00-99 format - 2 chars)\n    {defense_pct2} Moveset perfection for defense (in 00-99 format - 2 chars)\n    {attack_pct1}  Moveset perfection for attack (in 0-9 format - 1 char)\n    {defense_pct1} Moveset perfection for defense (in 0-9 format - 1 char)\n\n    # Special case: pokemon object.\n    # You can access any available pokemon info via it.\n    # Examples:\n    #   \'{pokemon.ivcp:.2%}\'             ->  \'47.00%\'\n    #   \'{pokemon.fast_attack}\'          ->  \'Wing Attack\'\n    #   \'{pokemon.fast_attack.type}\'     ->  \'Flying\'\n    #   \'{pokemon.fast_attack.dps:.2f}\'  ->  \'10.91\'\n    #   \'{pokemon.fast_attack.dps:.0f}\'  ->  \'11\'\n    #   \'{pokemon.charged_attack}\'       ->  \'Ominous Wind\'\n    {pokemon}   Pokemon instance (see inventory.py for class sources)\n\n\n    EXAMPLES:\n\n    1. "nickname_template": "{ivcp_pct}-{iv_pct}-{iv_ads}"\n\n    Golbat with IV (attack: 9, defense: 4 and stamina: 8) will result in:\n     \'48_46_9/4/8\'\n\n    2. "nickname_template": "{attack_code}{attack_pct1}{defense_pct1}{ivcp_pct1}{name}"\n\n    Same Golbat (with attacks Wing Attack & Ominous Wind) will have nickname:\n     \'Lh474Golbat\'\n\n    See /tests/nickname_test.py for more examples.\n    '

    def initialize(self):
        if False:
            return 10
        self.ignore_favorites = self.config.get('dont_nickname_favorite', DEFAULT_IGNORE_FAVORITES)
        self.good_attack_threshold = self.config.get('good_attack_threshold', DEFAULT_GOOD_ATTACK_THRESHOLD)
        self.template = self.config.get('nickname_template', DEFAULT_TEMPLATE)
        self.nickname_above_iv = self.config.get('nickname_above_iv', 0)
        self.nickname_wait_min = self.config.get('nickname_wait_min', DEFAULT_NICKNAME_WAIT_MIN)
        self.nickname_wait_max = self.config.get('nickname_wait_max', DEFAULT_NICKNAME_WAIT_MAX)
        self.translate = None
        locale = self.config.get('locale', 'en')
        if locale != 'en':
            fn = 'data/locales/{}.json'.format(locale)
            if os.path.isfile(fn):
                self.translate = json.load(open(fn))

    def work(self):
        if False:
            return 10
        '\n        Iterate over all user pokemons and nickname if needed\n        '
        for pokemon in pokemons().all():
            if not pokemon.is_favorite or not self.ignore_favorites:
                if pokemon.iv >= self.nickname_above_iv:
                    if self._nickname_pokemon(pokemon):
                        action_delay(self.nickname_wait_min, self.nickname_wait_max)

    def _localize(self, string):
        if False:
            return 10
        if self.translate and string in self.translate:
            return self.translate[string]
        else:
            return string

    def _nickname_pokemon(self, pokemon):
        if False:
            for i in range(10):
                print('nop')
        '\n        Nicknaming process\n        '
        instance_id = pokemon.unique_id
        if not instance_id:
            self.emit_event('api_error', formatted='Failed to get pokemon name, will not rename.')
            return False
        old_nickname = pokemon.nickname
        try:
            new_nickname = self._generate_new_nickname(pokemon, self.template)
        except KeyError as bad_key:
            self.emit_event('config_error', formatted='Unable to nickname {} due to bad template ({})'.format(old_nickname, bad_key))
            return False
        if pokemon.nickname_raw == new_nickname:
            return False
        request = self.bot.api.create_request()
        request.nickname_pokemon(pokemon_id=instance_id, nickname=new_nickname)
        response = request.call()
        sleep(1.2)
        try:
            result = reduce(dict.__getitem__, ['responses', 'NICKNAME_POKEMON'], response)['result']
        except KeyError:
            self.emit_event('api_error', formatted='Attempt to nickname received bad response from server.')
            return True
        if result == 0:
            self.emit_event('unset_pokemon_nickname', formatted='Pokemon {} nickname unset.'.format(old_nickname), data={'old_name': old_nickname})
            pokemon.update_nickname(new_nickname)
        elif result == 1:
            self.emit_event('rename_pokemon', formatted='*{} Renamed* to *{}*'.format(old_nickname, new_nickname), data={'old_name': old_nickname, 'current_name': new_nickname})
            pokemon.update_nickname(new_nickname)
        elif result == 2:
            self.emit_event('pokemon_nickname_invalid', formatted='Nickname {} is invalid'.format(new_nickname), data={'nickname': new_nickname})
        else:
            self.emit_event('api_error', formatted='Attempt to nickname received unexpected result from server ({}).'.format(result))
        return True

    def _generate_new_nickname(self, pokemon, template):
        if False:
            return 10
        '\n        New nickname generation\n        '
        template = re.sub('{[\\w_\\d]*', lambda x: x.group(0).lower(), template).strip()
        iv_attack = pokemon.iv_attack
        iv_defense = pokemon.iv_defense
        iv_stamina = pokemon.iv_stamina
        iv_list = [iv_attack, iv_defense, iv_stamina]
        iv_sum = sum(iv_list)
        iv_pct = iv_sum / 45.0
        base_attack = pokemon.static.base_attack
        base_defense = pokemon.static.base_defense
        base_stamina = pokemon.static.base_stamina
        attack = base_attack + iv_attack
        defense = base_defense + iv_defense
        stamina = base_stamina + iv_stamina
        fast_attack_char = self.attack_char(pokemon.fast_attack)
        charged_attack_char = self.attack_char(pokemon.charged_attack)
        attack_code = fast_attack_char + charged_attack_char
        moveset = pokemon.moveset
        pokemon.name = self._localize(pokemon.name)
        pokemon.name = pokemon.name.replace('Nidoran M', 'NidoranM')
        pokemon.name = pokemon.name.replace('Nidoran F', 'NidoranF')
        new_name = template.format(pokemon=pokemon, name=pokemon.name, id=int(pokemon.pokemon_id), cp=int(pokemon.cp), iv_attack=iv_attack, iv_defense=iv_defense, iv_stamina=iv_stamina, iv_ads='/'.join(map(str, iv_list)), iv_ads_hex=''.join(map(lambda x: format(x, 'X'), iv_list)), iv_sum=iv_sum, iv_pct='{:03.0f}'.format(iv_pct * 100), iv_pct2='{:02.0f}'.format(iv_pct * 99), iv_pct1=int(round(iv_pct * 9)), base_attack=base_attack, base_defense=base_defense, base_stamina=base_stamina, base_ads='/'.join(map(str, [base_attack, base_defense, base_stamina])), attack=attack, defense=defense, stamina=stamina, sum_ads='/'.join(map(str, [attack, defense, stamina])), ivcp_pct='{:03.0f}'.format(pokemon.ivcp * 100), ivcp_pct2='{:02.0f}'.format(pokemon.ivcp * 99), ivcp_pct1=int(round(pokemon.ivcp * 9)), fast_attack_char=fast_attack_char, charged_attack_char=charged_attack_char, attack_code=attack_code, attack_pct='{:03.0f}'.format(moveset.attack_perfection * 100), defense_pct='{:03.0f}'.format(moveset.defense_perfection * 100), attack_pct2='{:02.0f}'.format(moveset.attack_perfection * 99), defense_pct2='{:02.0f}'.format(moveset.defense_perfection * 99), attack_pct1=int(round(moveset.attack_perfection * 9)), defense_pct1=int(round(moveset.defense_perfection * 9)))
        if new_name == pokemon.name:
            new_name = ''
        return new_name[:MAXIMUM_NICKNAME_LENGTH]

    def attack_char(self, attack):
        if False:
            print('Hello World!')
        "\n        One character code for attack type\n        If attack is good then character is uppecased, otherwise lowercased\n\n        Type codes:\n\n        Bug: 'B'\n        Dark: 'K'\n        Dragon: 'D'\n        Electric: 'E'\n        Fairy: 'Y'\n        Fighting: 'T'\n        Fire: 'F'\n        Flying: 'L'\n        Ghost: 'H'\n        Grass: 'A'\n        Ground: 'G'\n        Ice: 'I'\n        Normal: 'N'\n        Poison: 'P'\n        Psychic: 'C'\n        Rock: 'R'\n        Steel: 'S'\n        Water: 'W'\n\n        it's an effective way to represent type with one character\n        if first char is unique - use it, in other case suitable substitute used\n        "
        char = attack.type.as_one_char.upper()
        if attack.rate_in_type < self.good_attack_threshold:
            char = char.lower()
        return char