import numpy as np
from urh.cythonext import util
from urh.signalprocessing.Message import Message

def auto_assign_participants(messages, participants):
    if False:
        print('Hello World!')
    '\n\n    :type messages: list of Message\n    :type participants: list of Participant\n    :return:\n    '
    if len(participants) == 0:
        return
    if len(participants) == 1:
        for message in messages:
            message.participant = participants[0]
        return
    for msg in filter(lambda m: m.participant is None, messages):
        src_address = msg.get_src_address_from_data()
        if src_address:
            try:
                msg.participant = next((p for p in participants if p.address_hex == src_address))
            except StopIteration:
                pass
    rssis = np.array([msg.rssi for msg in messages], dtype=np.float32)
    (min_rssi, max_rssi) = util.minmax(rssis)
    center_spacing = (max_rssi - min_rssi) / (len(participants) - 1)
    centers = [min_rssi + i * center_spacing for i in range(0, len(participants))]
    rssi_assigned_centers = []
    for rssi in rssis:
        center_index = np.argmin(np.abs(rssi - centers))
        rssi_assigned_centers.append(int(center_index))
    participants.sort(key=lambda participant: participant.relative_rssi)
    for (message, center_index) in zip(messages, rssi_assigned_centers):
        if message.participant is None:
            message.participant = participants[center_index]

def auto_assign_participant_addresses(messages, participants):
    if False:
        print('Hello World!')
    '\n\n    :type messages: list of Message\n    :type participants: list of Participant\n    :return:\n    '
    participants_without_address = [p for p in participants if not p.address_hex]
    if len(participants_without_address) == 0:
        return
    for msg in messages:
        if msg.participant in participants_without_address:
            src_address = msg.get_src_address_from_data()
            if src_address:
                participants_without_address.remove(msg.participant)
                msg.participant.address_hex = src_address