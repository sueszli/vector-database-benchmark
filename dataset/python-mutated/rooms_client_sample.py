import os
import sys
from datetime import datetime, timedelta
from azure.core.exceptions import HttpResponseError
from azure.communication.identity import CommunicationIdentityClient
from azure.communication.rooms import ParticipantRole, RoomsClient, RoomParticipant
sys.path.append('..')

class RoomsSample(object):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.connection_string = os.getenv('COMMUNICATION_CONNECTION_STRING_ROOMS')
        self.rooms_client = RoomsClient.from_connection_string(self.connection_string)
        self.identity_client = CommunicationIdentityClient.from_connection_string(self.connection_string)
        self.rooms = []
        self.participant_1 = RoomParticipant(communication_identifier=self.identity_client.create_user(), role=ParticipantRole.PRESENTER)
        self.participant_2 = RoomParticipant(communication_identifier=self.identity_client.create_user(), role=ParticipantRole.CONSUMER)

    def tearDown(self):
        if False:
            print('Hello World!')
        self.delete_room_all_rooms()

    def create_single_room(self):
        if False:
            return 10
        valid_from = datetime.now()
        valid_until = valid_from + timedelta(weeks=4)
        participants = [self.participant_1]
        try:
            create_room_response = self.rooms_client.create_room(valid_from=valid_from, valid_until=valid_until, participants=participants)
            self.printRoom(response=create_room_response)
            self.rooms.append(create_room_response.id)
        except HttpResponseError as ex:
            print(ex)

    def create_single_room_with_default_attributes(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            create_room_response = self.rooms_client.create_room()
            self.printRoom(response=create_room_response)
            self.rooms.append(create_room_response.id)
        except HttpResponseError as ex:
            print(ex)

    def create_room_with_pstn_attribute(self):
        if False:
            print('Hello World!')
        valid_from = datetime.now()
        valid_until = valid_from + timedelta(weeks=4)
        participants = [self.participant_1]
        pstn_dial_out_enabled = True
        try:
            create_room_response = self.rooms_client.create_room(valid_from=valid_from, valid_until=valid_until, participants=participants, pstn_dial_out_enabled=pstn_dial_out_enabled)
            self.printRoom(response=create_room_response)
            self.rooms.append(create_room_response.id)
        except HttpResponseError as ex:
            print(ex)

    def update_single_room(self, room_id):
        if False:
            for i in range(10):
                print('nop')
        valid_from = datetime.now()
        valid_until = valid_from + timedelta(weeks=7)
        try:
            update_room_response = self.rooms_client.update_room(room_id=room_id, valid_from=valid_from, valid_until=valid_until)
            self.printRoom(response=update_room_response)
        except HttpResponseError as ex:
            print(ex)

    def update_room_with_pstn_attribute(self, room_id):
        if False:
            while True:
                i = 10
        valid_from = datetime.now()
        valid_until = valid_from + timedelta(weeks=7)
        pstn_dial_out_enabled = True
        try:
            update_room_response = self.rooms_client.update_room(room_id=room_id, valid_from=valid_from, valid_until=valid_until, pstn_dial_out_enabled=pstn_dial_out_enabled)
            self.printRoom(response=update_room_response)
        except HttpResponseError as ex:
            print(ex)

    def add_or_update_participants(self, room_id):
        if False:
            for i in range(10):
                print('nop')
        self.participant_1.role = ParticipantRole.ATTENDEE
        participants = [self.participant_1, self.participant_2]
        try:
            self.rooms_client.add_or_update_participants(room_id=room_id, participants=participants)
        except HttpResponseError as ex:
            print(ex)

    def list_participants(self, room_id):
        if False:
            print('Hello World!')
        try:
            get_participants_response = self.rooms_client.list_participants(room_id=room_id)
            print('participants: \n', self.convert_participant_list_to_string(get_participants_response))
        except HttpResponseError as ex:
            print(ex)

    def remove_participants(self, room_id):
        if False:
            while True:
                i = 10
        participants = [self.participant_1.communication_identifier]
        try:
            self.rooms_client.remove_participants(room_id=room_id, participants=participants)
        except HttpResponseError as ex:
            print(ex)

    def delete_room_all_rooms(self):
        if False:
            while True:
                i = 10
        for room in self.rooms:
            print('deleting: ', room)
            self.rooms_client.delete_room(room_id=room)

    def get_room(self, room_id):
        if False:
            i = 10
            return i + 15
        try:
            get_room_response = self.rooms_client.get_room(room_id=room_id)
            self.printRoom(response=get_room_response)
        except HttpResponseError as ex:
            print(ex)

    def printRoom(self, response):
        if False:
            return 10
        print('room_id: ', response.id)
        print('created_at: ', response.created_at)
        print('valid_from: ', response.valid_from)
        print('valid_until: ', response.valid_until)

    def convert_participant_list_to_string(self, participants):
        if False:
            i = 10
            return i + 15
        result = ''
        for p in participants:
            result += 'id: {}\n role: {}\n'.format(p.communication_identifier.properties['id'], p.role)
        return result
if __name__ == '__main__':
    sample = RoomsSample()
    sample.setUp()
    sample.create_single_room()
    sample.create_single_room_with_default_attributes()
    if len(sample.rooms) > 0:
        sample.get_room(room_id=sample.rooms[0])
        sample.update_single_room(room_id=sample.rooms[0])
        sample.add_or_update_participants(room_id=sample.rooms[0])
        sample.list_participants(room_id=sample.rooms[0])
        sample.remove_participants(room_id=sample.rooms[0])
        sample.get_room(room_id=sample.rooms[0])
    sample.tearDown()