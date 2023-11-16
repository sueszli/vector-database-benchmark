from __future__ import unicode_literals
from __future__ import absolute_import
from datetime import datetime, timedelta
import sys
import time
import random
from random import uniform
from collections import Counter
from pgoapi.utilities import f2i
from pokemongo_bot import inventory
from pokemongo_bot.inventory import player
from pokemongo_bot.constants import Constants
from pokemongo_bot.human_behaviour import action_delay, sleep
from pokemongo_bot.worker_result import WorkerResult
from pokemongo_bot.base_task import BaseTask
from pokemongo_bot import inventory
from .utils import distance, format_time, fort_details, format_dist
from pokemongo_bot.tree_config_builder import ConfigException
from pokemongo_bot.walkers.walker_factory import walker_factory
from pokemongo_bot.inventory import Pokemons
from sys import stdout
GYM_DETAIL_RESULT_SUCCESS = 1
GYM_DETAIL_RESULT_OUT_OF_RANGE = 2
GYM_DETAIL_RESULT_UNSET = 0
TEAM_NOT_SET = 0
TEAM_BLUE = 1
TEAM_RED = 2
TEAM_YELLOW = 3
TEAMS = {0: 'Not Set', 1: 'Mystic', 2: 'Valor', 3: 'Instinct'}
ITEM_RAZZBERRY = 701
ITEM_NANABBERRY = 703
ITEM_PINAPBERRY = 705

class GymPokemon(BaseTask):
    SUPPORTED_TASK_API_VERSION = 1

    def __init__(self, bot, config):
        if False:
            for i in range(10):
                print('nop')
        super(GymPokemon, self).__init__(bot, config)

    def initialize(self):
        if False:
            for i in range(10):
                print('nop')
        self.next_update = datetime.now() + timedelta(0, 10)
        self.order_by = self.config.get('order_by', 'cp')
        self.enabled = self.config.get('enabled', False)
        self.min_interval = self.config.get('min_interval', 360)
        self.min_recheck = self.config.get('min_recheck', 30)
        self.max_recheck = self.config.get('max_recheck', 120)
        self.take_at_most = self.config.get('take_at_most', 20)
        if self.take_at_most > 20:
            self.logger.warning('We cannot take more than 20 gyms!')
            self.take_at_most = 20
        self.leave_at_least_spots = self.config.get('leave_at_least_spots', 0)
        if self.leave_at_least_spots > 4:
            self.logger.warning('There are only 6 spots in a gym, when we drop a Pokemon in that would leave 5 spots! Setting leave open spots reset to 4!')
            self.leave_at_least_spots = 4
        self.chain_fill_gyms = self.config.get('chain_fill_gyms', True)
        self.ignore_max_cp_pokemon = self.config.get('allow_above_cp', ['Blissey'])
        self.never_place = self.config.get('never_place', [])
        self.pick_random_pokemon = self.config.get('pick_random_pokemon', True)
        self.can_be_disabled_by_catch_limter = self.config.get('can_be_disabled_by_catch_limter', False)
        self.raid = self.config.get('raid', False)
        self.raid_levels = self.config.get('raid_levels', [])
        self.raid_only = self.config.get('raid_only', [])
        self.raid_min_players = self.config.get('raid_min_players', [])
        self.wait_raid_start_mins = self.config.get('wait_raid_start_mins', 0)
        self.wait_min_players_mins = self.config.get('wait_min_players_mins', 0)
        self.do_not_raid_last_x_mins = self.config.get('do_not_raid_last_x_mins', 10)
        self.use_paid_tickets = self.config.get('use_paid_tickets', False)
        items_inventory = inventory.items()
        self.free_raid_tickets = items_inventory.get(1401).count
        self.paid_raid_tickets = items_inventory.get(1402).count
        if self.use_paid_tickets == False:
            self.paid_raid_tickets = 0
        self.found_raid = False
        self.raided_gyms = []
        self.recheck = datetime.now()
        self.walker = self.config.get('walker', 'StepWalker')
        self.destination = None
        self.recent_gyms = []
        self.pokemons = []
        self.fort_pokemons = []
        self.expire_recent = 10
        self.next_expire = None
        self.dropped_gyms = []
        self.blacklist = []
        self.check_interval = 0
        self.gyms = []
        self.raid_gyms = dict()
        self.bot.event_manager.register_event('gym_error')
        self.bot.event_manager.register_event('fed_pokemon')
        self.bot.event_manager.register_event('gym_full')
        self.bot.event_manager.register_event('deployed_pokemon')
        try:
            self.team = self.bot.player_data['team']
        except KeyError:
            self.team = TEAM_NOT_SET
            if self.enabled:
                self.emit_event('gym_error', formatted='You have no team selected, so the module GymPokemon should be disabled')

    def should_run(self):
        if False:
            i = 10
            return i + 15
        return player()._level >= 5 and len(self.pokemons) > 0 and (self.team > TEAM_NOT_SET)

    def display_fort_pokemon(self):
        if False:
            i = 10
            return i + 15
        if len(self.fort_pokemons) == 0:
            return
        self.logger.info('We currently have %s Pokemon in Gym(s):' % len(self.fort_pokemons))
        for pokemon in self.fort_pokemons:
            lat = self.bot.position[0:2][0]
            lng = self.bot.position[0:2][1]
            self.logger.info('%s (%s CP)' % (pokemon.name, pokemon.cp))

    def work(self):
        if False:
            while True:
                i = 10
        if not self.enabled:
            return WorkerResult.SUCCESS
        if self.bot.catch_disabled and self.can_be_disabled_by_catch_limter:
            if self.destination is not None:
                self.destination = None
            if not hasattr(self.bot, 'gym_pokemon_disabled_global_warning') or (hasattr(self.bot, 'gym_pokemon_disabled_global_warning') and (not self.bot.gym_pokemon_disabled_global_warning)):
                self.logger.info('All gym tasks are currently disabled until {}. Gym function will resume when catching tasks are re-enabled'.format(self.bot.catch_resume_at.strftime('%H:%M:%S')))
            self.bot.gym_pokemon_disabled_global_warning = True
            return WorkerResult.SUCCESS
        else:
            self.bot.gym_pokemon_disabled_global_warning = False
        self.pokemons = inventory.pokemons().all()
        self.fort_pokemons = [p for p in self.pokemons if p.in_fort]
        self.pokemons = [p for p in self.pokemons if not p.in_fort]
        self.dropped_gyms = []
        for pokemon in self.fort_pokemons:
            self.dropped_gyms.append(pokemon.fort_id)
        if self._should_print():
            self.display_fort_pokemon()
            self._compute_next_update()
        if not self.enabled:
            return WorkerResult.SUCCESS
        if self.bot.softban:
            return WorkerResult.SUCCESS
        if len(self.fort_pokemons) >= self.take_at_most:
            if self._should_print():
                self.logger.info('We have a max of %s Pokemon in gyms.' % self.take_at_most)
            return WorkerResult.SUCCESS
        if not self.should_run():
            return WorkerResult.SUCCESS
        if self.destination is None:
            self.check_close_gym()
        if self.destination is None:
            self.determin_new_destination()
        if self.destination is not None:
            result = self.move_to_destination()
            return result
        if hasattr(self.bot, 'hunter_locked_target') and self.bot.hunter_locked_target is not None:
            return WorkerResult.SUCCESS
        return WorkerResult.SUCCESS

    def check_raid(self):
        if False:
            i = 10
            return i + 15
        raids = self.get_gyms_in_range(raids=True)
        for raid in raids:
            if 'raid_info' in raid:
                for level in self.raid_levels:
                    if level == raid['raid_info']['raid_level']:
                        org_time = int(raid['raid_info']['raid_battle_ms']) / 1000.0
                        raid_start_time = datetime.fromtimestamp(org_time)
                        org_time = int(raid['raid_info']['raid_end_ms']) / 1000.0
                        raid_end_time = datetime.fromtimestamp(org_time)
                        if raid_start_time > datetime.now():
                            timediff = raid_start_time - datetime.now()
                            results = divmod(timediff.days * 86400 + timediff.seconds, 60)
                            print('There is a raid, raid level: ' + format(raid['raid_info']['raid_level']) + ' ID: ' + format(raid['id']))
                            print('Raid will be starting in : ' + format(results[0]) + ' mins')
                            print('Raw Start Time: ' + format(raid['raid_info']['raid_battle_ms']) + '\n')
                        else:
                            timediff = raid_end_time - datetime.now()
                            results = divmod(timediff.days * 86400 + timediff.seconds, 60)
                            print('There is a raid, raid level: ' + format(raid['raid_info']['raid_level']) + ' ID: ' + format(raid['id']))
                            print('Raid has started, ending in: ' + format(results[0]) + ' mins')
                            print('Raw End Time: ' + format(raid['raid_info']['raid_end_ms']) + '\n')

    def check_close_gym(self):
        if False:
            i = 10
            return i + 15
        close_gyms = self.get_gyms_in_range()
        close_gyms = filter(lambda gym: gym['id'] not in self.raid_gyms, close_gyms)
        if len(close_gyms) > 0:
            for gym in close_gyms:
                if gym['id'] in self.dropped_gyms:
                    continue
                gym_details = self.get_gym_details(gym)
                if gym_details:
                    pokes = self._get_pokemons_in_gym(gym_details)
                    if len(pokes) == 6:
                        continue
                    if 'enabled' in gym:
                        if not gym['enabled']:
                            continue
                    if 'owned_by_team' in gym:
                        if gym['owned_by_team'] == self.team:
                            if 'gym_display' in gym:
                                display = gym['gym_display']
                                if 'slots_available' in display:
                                    self.logger.info('Gym has %s open spots!' % display['slots_available'])
                                    if display['slots_available'] > 0 and gym['id'] not in self.dropped_gyms:
                                        self.logger.info('Dropping pokemon in %s' % gym_details['name'])
                                        self.drop_pokemon_in_gym(gym, pokes)
                                        if self.destination is not None and gym['id'] == self.destination['id']:
                                            self.destination = None
                                        return WorkerResult.SUCCESS
                    else:
                        self.logger.info('Neutral gym? %s' % gym)
                        self.logger.info('Dropping pokemon in %s' % gym_details['name'])
                        self.drop_pokemon_in_gym(gym, [])
                        if self.destination is not None and gym['id'] == self.destination['id']:
                            self.destination = None
                        return WorkerResult.SUCCESS

    def determin_new_destination(self):
        if False:
            i = 10
            return i + 15
        gyms = self.get_gyms(get_raids=False)
        if len(gyms) == 0:
            if len(self.recent_gyms) == 0 and self._should_print():
                self.logger.info('No Gyms in range to scan!')
            return WorkerResult.SUCCESS
        self.logger.info('Inspecting %s gyms.' % len(gyms))
        self.logger.info('Recent gyms: %s' % len(self.recent_gyms))
        self.logger.info('Active raid gyms: %s' % len(self.raid_gyms))
        teams = []
        for gym in gyms:
            self.recent_gyms.append(gym['id'])
            if 'enabled' in gym:
                if not gym['enabled']:
                    continue
            if 'owned_by_team' in gym:
                if gym['owned_by_team'] == 1:
                    teams.append('Mystic')
                elif gym['owned_by_team'] == 2:
                    teams.append('Valor')
                elif gym['owned_by_team'] == 3:
                    teams.append('Instinct')
                if gym['owned_by_team'] == self.team:
                    if 'gym_display' in gym:
                        display = gym['gym_display']
                        if 'slots_available' in display:
                            if self.leave_at_least_spots > 0:
                                if display['slots_available'] > self.leave_at_least_spots:
                                    self.logger.info('Gym has %s open spots!' % display['slots_available'])
                                    self.destination = gym
                                    break
                                else:
                                    self.logger.info("Gym has %s open spots, but we don't drop Pokemon in it because that would leave less than %s open spots" % (display['slots_available'], self.leave_at_least_spots))
                            else:
                                self.logger.info('Gym has %s open spots!' % display['slots_available'])
                                self.destination = gym
                                break
            else:
                self.destination = gym
                break
        if len(teams) > 0:
            count_teams = Counter(teams)
            self.logger.info('Gym Teams %s', ', '.join(('{}({})'.format(key, val) for (key, val) in count_teams.items())))
        self.found_raid = False
        pokemon_in_raid = None
        if self.raid and (self.free_raid_tickets > 0 or self.paid_raid_tickets > 0) and (self.destination is None):
            self.logger.info('Checking for eligable raids')
            self.recent_gyms = []
            gyms = self.get_gyms(get_raids=True)
            for gym in gyms:
                self.recent_gyms.append(gym['id'])
                if 'raid_info' in gym:
                    for level in self.raid_levels:
                        if level == gym['raid_info']['raid_level']:
                            org_time = int(gym['raid_info']['raid_battle_ms']) / 1000.0
                            raid_start_time = datetime.fromtimestamp(org_time)
                            org_time = int(gym['raid_info']['raid_end_ms']) / 1000.0
                            raid_end_time = datetime.fromtimestamp(org_time)
                            if raid_start_time < datetime.now():
                                timediff = raid_end_time - datetime.now()
                                results = divmod(timediff.days * 86400 + timediff.seconds, 60)
                                details = fort_details(self.bot, gym['id'], gym['latitude'], gym['longitude'])
                                gym_name = details.get('name', 'Unknown')
                                raid_pokemon_id = gym['raid_info']['raid_pokemon']['pokemon_id']
                                raid_pokemon_name = Pokemons.name_for(raid_pokemon_id)
                                pokemon_in_raid = [p for p in self.raid_only if p in raid_pokemon_name]
                                if results[0] >= self.do_not_raid_last_x_mins and (len(pokemon_in_raid) > 0 or len(self.raid_only) == 0):
                                    self.logger.info('There is an on-going raid. Raid level: ' + format(gym['raid_info']['raid_level']) + ' Name: ' + gym_name + ' Ending in: ' + format(results[0]) + ' mins')
                                    self.logger.info('Raid Boss: ' + format(raid_pokemon_name))
                                    self.logger.info('We have enough time for raid!')
                                    self.found_raid = True
                                    self.destination = gym
                                    break
                            else:
                                timediff = raid_start_time - datetime.now()
                                results = divmod(timediff.days * 86400 + timediff.seconds, 60)
                                details = fort_details(self.bot, gym['id'], gym['latitude'], gym['longitude'])
                                gym_name = details.get('name', 'Unknown')
                                if results[0] <= self.wait_raid_start_mins:
                                    self.logger.info('A raid is starting soon. Raid level: ' + format(gym['raid_info']['raid_level']) + ' Name: ' + gym_name + ' Raid starting in ' + format(results[0]) + ' mins')
                                    self.logger.info('It is within ' + format(self.wait_raid_start_mins) + ' mins')
                                    self.found_raid = True
                                    self.destination = gym
                                    break
                    if self.found_raid:
                        break
                    else:
                        self.logger.info('No suitable raids available')

    def move_to_destination(self):
        if False:
            print('Hello World!')
        if self.check_interval >= 4 and (not self.found_raid):
            self.check_interval = 0
            gyms = self.get_gyms()
            for g in gyms:
                if g['id'] == self.destination['id']:
                    if 'owned_by_team' in g and g['owned_by_team'] is not self.team:
                        self.logger.info('Damn! Team %s took gym before we arrived!' % TEAMS[g['owned_by_team']])
                        self.destination = None
                        return WorkerResult.SUCCESS
                    break
        elif not self.found_raid:
            self.check_interval += 1
        unit = self.bot.config.distance_unit
        lat = self.destination['latitude']
        lng = self.destination['longitude']
        details = fort_details(self.bot, self.destination['id'], lat, lng)
        gym_name = details.get('name', 'Unknown')
        dist = distance(self.bot.position[0], self.bot.position[1], lat, lng)
        noised_dist = distance(self.bot.noised_position[0], self.bot.noised_position[1], lat, lng)
        moving = noised_dist > Constants.MAX_DISTANCE_FORT_IS_REACHABLE if self.bot.config.replicate_gps_xy_noise else dist > Constants.MAX_DISTANCE_FORT_IS_REACHABLE
        if moving:
            fort_event_data = {'fort_name': u'{}'.format(gym_name), 'distance': format_dist(dist, unit)}
            if not self.found_raid:
                self.emit_event('moving_to_fort', formatted='Moving towards open Gym {fort_name} - {distance}', data=fort_event_data)
            else:
                self.emit_event('moving_to_fort', formatted='Moving towards raid Gym {fort_name} - {distance}', data=fort_event_data)
            step_walker = walker_factory(self.walker, self.bot, lat, lng)
            if not step_walker.step():
                return WorkerResult.RUNNING
            else:
                return WorkerResult.SUCCESS
        else:
            self.emit_event('arrived_at_fort', formatted='Arrived at Gym %s.' % gym_name)
            gym_details = self.get_gym_details(self.destination)
            if not self.found_raid:
                current_pokemons = self._get_pokemons_in_gym(gym_details)
                self.drop_pokemon_in_gym(self.destination, current_pokemons)
                self.destination = None
                if len(self.fort_pokemons) >= self.take_at_most:
                    self.logger.info('We have a max of %s Pokemon in gyms.' % self.take_at_most)
                    return WorkerResult.SUCCESS
                elif self.chain_fill_gyms:
                    self.determin_new_destination()
                    if self.destination is None:
                        return WorkerResult.SUCCESS
                    else:
                        return WorkerResult.RUNNING
                else:
                    return WorkerResult.SUCCESS
            else:
                gym_info = gym_details.get('gym_status_and_defenders', None)
                if gym_info is not None:
                    pokemon_fort_proto = gym_info.get('pokemon_fort_proto')
                    raid_info = pokemon_fort_proto.get('raid_info')
                    raid_level = raid_info['raid_level']
                    raid_seed = raid_info['raid_seed']
                    raid_end_ms = raid_info['raid_end_ms']
                    raid_battle_ms = raid_info['raid_battle_ms']
                    raid_starts = datetime.fromtimestamp(int(raid_info['raid_battle_ms']) / 1000.0)
                    raid_ends = datetime.fromtimestamp(int(raid_info['raid_end_ms']) / 1000.0)
                    self.logger.info('Raid starts: %s' % raid_starts.strftime('%Y-%m-%d %H:%M:%S.%f'))
                    self.logger.info('Raid ends: %s' % raid_ends.strftime('%Y-%m-%d %H:%M:%S.%f'))
                    if raid_starts < datetime.now():
                        timediff = raid_ends - datetime.now()
                        results = divmod(timediff.days * 86400 + timediff.seconds, 60)
                        if results[0] >= self.do_not_raid_last_x_mins:
                            a = 0
                    else:
                        self.logger.info('Raid has not begun yet!')
                self.logger.info('Assume we are done with raid')
                print('id: ' + format(gym_info['pokemon_fort_proto']['id']))
                if gym_info['pokemon_fort_proto']['id']:
                    self.raided_gyms.append(gym_info['pokemon_fort_proto']['id'])
                print('dropped_gyms: ' + format(self.raided_gyms))
                self.destination = None
                self.found_raid = False
                return WorkerResult.SUCCESS

    def get_gym_details(self, gym):
        if False:
            while True:
                i = 10
        lat = gym['latitude']
        lng = gym['longitude']
        in_reach = False
        if self.bot.config.replicate_gps_xy_noise:
            if distance(self.bot.noised_position[0], self.bot.noised_position[1], gym['latitude'], gym['longitude']) <= Constants.MAX_DISTANCE_FORT_IS_REACHABLE:
                in_reach = True
        elif distance(self.bot.position[0], self.bot.position[1], gym['latitude'], gym['longitude']) <= Constants.MAX_DISTANCE_FORT_IS_REACHABLE:
            in_reach = True
        if in_reach:
            request = self.bot.api.create_request()
            request.gym_get_info(gym_id=gym['id'], gym_lat_degrees=lat, gym_lng_degrees=lng, player_lat_degrees=self.bot.position[0], player_lng_degrees=self.bot.position[1])
            response_dict = request.call()
            if 'responses' in response_dict and 'GYM_GET_INFO' in response_dict['responses']:
                details = response_dict['responses']['GYM_GET_INFO']
                return details
        else:
            return False

    def _get_pokemons_in_gym(self, gym_details):
        if False:
            i = 10
            return i + 15
        pokemon_names = []
        gym_info = gym_details.get('gym_status_and_defenders', None)
        if gym_info:
            defenders = gym_info.get('gym_defender', [])
            for defender in defenders:
                motivated_pokemon = defender.get('motivated_pokemon')
                pokemon_info = motivated_pokemon.get('pokemon')
                pokemon_id = pokemon_info.get('pokemon_id')
                pokemon_names.append(Pokemons.name_for(pokemon_id))
        return pokemon_names

    def drop_pokemon_in_gym(self, gym, current_pokemons):
        if False:
            for i in range(10):
                print('nop')
        self.pokemons = inventory.pokemons().all()
        self.fort_pokemons = [p for p in self.pokemons if p.in_fort]
        self.pokemons = [p for p in self.pokemons if not p.in_fort]
        close_gyms = self.get_gyms_in_range()
        empty_gym = False
        for pokemon in self.fort_pokemons:
            if pokemon.fort_id == gym['id']:
                self.logger.info('We are already in this gym!')
                if pokemon.fort_id not in self.dropped_gyms:
                    self.dropped_gyms.append(pokemon.fort_id)
                self.recent_gyms.append(gym['id'])
                return WorkerResult.SUCCESS
        for g in close_gyms:
            if g['id'] == gym['id']:
                if 'owned_by_team' in g:
                    self.logger.info('Expecting team: %s it is: %s' % (self.bot.player_data['team'], g['owned_by_team']))
                    if g['owned_by_team'] is not self.team:
                        self.logger.info("Can't drop in a enemy gym!")
                        self.recent_gyms.append(gym['id'])
                        return WorkerResult.SUCCESS
                else:
                    gym_details = self.get_gym_details(gym)
                    empty_gym = True
                    if not gym_details or gym_details == {}:
                        self.logger.info('No details for this Gym? Blacklisting!')
                        self.blacklist.append(gym['id'])
                        return WorkerResult.SUCCESS
        if 'raid_info' in gym:
            raid_info = gym['raid_info']
            raid_starts = datetime.fromtimestamp(int(raid_info['raid_battle_ms']) / 1000.0)
            raid_ends = datetime.fromtimestamp(int(raid_info['raid_end_ms']) / 1000.0)
            self.logger.info('Raid starts: %s' % raid_starts.strftime('%Y-%m-%d %H:%M:%S.%f'))
            self.logger.info('Raid ends: %s' % raid_ends.strftime('%Y-%m-%d %H:%M:%S.%f'))
            t = datetime.today()
            if raid_starts < datetime.now():
                self.logger.info('Active raid?')
                if raid_ends < datetime.now():
                    self.logger.info('No need to wait.')
                elif (raid_ends - t).seconds > 600:
                    self.logger.info('Need to wait more than 10 minutes, skipping')
                    self.destination = None
                    self.recent_gyms.append(gym['id'])
                    self.raid_gyms[gym['id']] = raid_ends
                    return WorkerResult.SUCCESS
                else:
                    first_time = False
                    while raid_ends > datetime.now():
                        raid_ending = (raid_ends - datetime.today()).seconds
                        (sleep_m, sleep_s) = divmod(raid_ending, 60)
                        (sleep_h, sleep_m) = divmod(sleep_m, 60)
                        sleep_hms = '%02d:%02d:%02d' % (sleep_h, sleep_m, sleep_s)
                        if not first_time:
                            stdout.write('\x1b[1A\x1b[0K\r')
                            stdout.flush()
                        first_time = True
                        self.logger.info('Waiting for %s for raid to end...' % sleep_hms)
                        if raid_ending > 20:
                            sleep(20)
                        else:
                            sleep(raid_ending)
                            break
            else:
                self.logger.info('Raid has not begun yet!')
        if 'same_team_deploy_lockout_end_ms' in gym:
            org_time = int(gym['same_team_deploy_lockout_end_ms']) / 1000.0
            lockout_time = datetime.fromtimestamp(org_time)
            t = datetime.today()
            if lockout_time > datetime.now():
                self.logger.info('Lockout time: %s' % lockout_time.strftime('%Y-%m-%d %H:%M:%S.%f'))
                first_time = False
                while lockout_time > datetime.now():
                    lockout_ending = (lockout_time - datetime.today()).seconds
                    (sleep_m, sleep_s) = divmod(lockout_ending, 60)
                    (sleep_h, sleep_m) = divmod(sleep_m, 60)
                    sleep_hms = '%02d:%02d:%02d' % (sleep_h, sleep_m, sleep_s)
                    if not first_time:
                        stdout.write('\x1b[1A\x1b[0K\r')
                        stdout.flush()
                    first_time = True
                    self.logger.info('Waiting for %s deployment lockout to end...' % sleep_hms)
                    if lockout_ending > 40:
                        sleep(40)
                        break
                    else:
                        sleep(lockout_ending)
                        break
        gym_details = self.get_gym_details(gym)
        slots_available = 6 - len(gym_details['gym_status_and_defenders']['gym_defender'])
        if slots_available <= self.leave_at_least_spots:
            self.logger.info("Gym has %s open slots, but we don't drop Pokemon in it because that would leave less than %s open spots" % (slots_available, self.leave_at_least_spots))
            return WorkerResult.ERROR
        fort_pokemon = self._get_best_pokemon(current_pokemons)
        pokemon_id = fort_pokemon.unique_id
        request = self.bot.api.create_request()
        request.gym_deploy(fort_id=gym['id'], pokemon_id=pokemon_id, player_latitude=f2i(self.bot.position[0]), player_longitude=f2i(self.bot.position[1]))
        response_dict = request.call()
        if 'responses' in response_dict and 'GYM_DEPLOY' in response_dict['responses']:
            deploy = response_dict['responses']['GYM_DEPLOY']
            result = response_dict.get('status_code', -1)
            self.recent_gyms.append(gym['id'])
            if result == 1:
                self.dropped_gyms.append(gym['id'])
                self.fort_pokemons.append(fort_pokemon)
                gym_details = self.get_gym_details(gym)
                self.emit_event('deployed_pokemon', formatted='We deployed %s (%s CP) in the gym %s!!' % (fort_pokemon.name, fort_pokemon.cp, gym_details['name']), data={'gym_id': gym['id'], 'pokemon_id': pokemon_id})
                return WorkerResult.SUCCESS
            elif result == 2:
                self.logger.info('ERROR_ALREADY_HAS_POKEMON_ON_FORT')
                self.dropped_gyms.append(gym['id'])
                return WorkerResult.ERROR
            elif result == 3:
                self.logger.info('ERROR_OPPOSING_TEAM_OWNS_FORT')
                return WorkerResult.ERROR
            elif result == 4:
                self.logger.info('ERROR_FORT_IS_FULL')
                return WorkerResult.ERROR
            elif result == 5:
                self.logger.info('ERROR_NOT_IN_RANGE')
                return WorkerResult.ERROR
            elif result == 6:
                self.logger.info('ERROR_PLAYER_HAS_NO_TEAM')
                return WorkerResult.ERROR
            elif result == 7:
                self.logger.info('ERROR_POKEMON_NOT_FULL_HP')
                return WorkerResult.ERROR
            elif result == 8:
                self.logger.info('ERROR_PLAYER_BELOW_MINIMUM_LEVEL')
                return WorkerResult.ERROR
            elif result == 8:
                self.logger.info('ERROR_POKEMON_IS_BUDDY')
                return WorkerResult.ERROR

    def get_gyms(self, skip_recent_filter=False, get_raids=False):
        if False:
            print('Hello World!')
        if len(self.gyms) == 0:
            self.gyms = self.bot.get_gyms(order_by_distance=True)
        if self._should_recheck():
            self.gyms = self.bot.get_gyms(order_by_distance=True)
            self._compute_next_recheck()
        if self._should_expire():
            self.recent_gyms = []
            self._compute_next_expire()
        for gym_id in list(self.raid_gyms.keys()):
            if self.raid_gyms[gym_id] < datetime.now():
                self.logger.info('Raid at %s ended (%s)' % (gym_id, self.raid_gyms[gym_id]))
                del self.raid_gyms[gym_id]
        gyms = []
        gyms = filter(lambda gym: gym['id'] not in self.recent_gyms, self.gyms)
        gyms = filter(lambda gym: gym['id'] not in self.blacklist, gyms)
        gyms = filter(lambda gym: gym['id'] not in self.dropped_gyms, gyms)
        gyms = filter(lambda gym: gym['id'] not in self.raided_gyms, gyms)
        if not get_raids:
            gyms = filter(lambda gym: gym['id'] not in self.raid_gyms, gyms)
        gyms.sort(key=lambda x: distance(self.bot.position[0], self.bot.position[1], x['latitude'], x['longitude']))
        return gyms

    def get_gyms_in_range(self, raids=False):
        if False:
            print('Hello World!')
        gyms = self.get_gyms(get_raids=raids)
        if self.bot.config.replicate_gps_xy_noise:
            gyms = filter(lambda fort: distance(self.bot.noised_position[0], self.bot.noised_position[1], fort['latitude'], fort['longitude']) <= Constants.MAX_DISTANCE_FORT_IS_REACHABLE, self.gyms)
        else:
            gyms = filter(lambda fort: distance(self.bot.position[0], self.bot.position[1], fort['latitude'], fort['longitude']) <= Constants.MAX_DISTANCE_FORT_IS_REACHABLE, self.gyms)
        return gyms

    def _should_print(self):
        if False:
            for i in range(10):
                print('nop')
        return self.next_update is None or datetime.now() >= self.next_update

    def _should_expire(self):
        if False:
            while True:
                i = 10
        return self.next_expire is None or datetime.now() >= self.next_expire

    def _compute_next_expire(self):
        if False:
            for i in range(10):
                print('nop')
        self.next_expire = datetime.now() + timedelta(seconds=300)

    def _compute_next_recheck(self):
        if False:
            i = 10
            return i + 15
        wait = uniform(self.min_recheck, self.max_recheck)
        self.recheck = datetime.now() + timedelta(seconds=wait)

    def _should_recheck(self):
        if False:
            for i in range(10):
                print('nop')
        return self.recheck is None or datetime.now() >= self.recheck

    def _compute_next_update(self):
        if False:
            return 10
        '\n        Computes the next update datetime based on the minimum update interval.\n        :return: Nothing.\n        :rtype: None\n        '
        self.next_update = datetime.now() + timedelta(seconds=self.min_interval)

    def _get_best_pokemon(self, current_pokemons):
        if False:
            return 10

        def get_poke_info(info, pokemon):
            if False:
                return 10
            poke_info = {'cp': pokemon.cp, 'iv': pokemon.iv, 'ivcp': pokemon.ivcp, 'ncp': pokemon.cp_percent, 'level': pokemon.level, 'hp': pokemon.hp, 'dps': pokemon.moveset.dps}
            if info not in poke_info:
                raise ConfigException("order by {}' isn't available".format(self.order_by))
            return poke_info[info]
        legendaries = ['Lugia', 'Zapdos', 'HoOh', 'Celebi', 'Articuno', 'Moltres', 'Mewtwo', 'Mew']
        possible_pokemons = [p for p in self.pokemons if not p.name in current_pokemons]
        possible_pokemons = [p for p in possible_pokemons if p.cp < 3000 and p.name not in self.ignore_max_cp_pokemon]
        possible_pokemons = [p for p in possible_pokemons if not p.is_bad]
        possible_pokemons = [p for p in possible_pokemons if not p.name in legendaries]
        possible_pokemons = [p for p in possible_pokemons if p.name not in self.never_place]
        possible_pokemons = [p for p in possible_pokemons if p.hp == p.hp_max]
        possible_pokemons = [p for p in possible_pokemons if not p.in_fort]
        pokemons_ordered = sorted(possible_pokemons, key=lambda x: get_poke_info(self.order_by, x), reverse=True)
        pokemons_ordered = pokemons_ordered[0:20]
        if self.pick_random_pokemon:
            random.shuffle(pokemons_ordered)
        return pokemons_ordered[0]