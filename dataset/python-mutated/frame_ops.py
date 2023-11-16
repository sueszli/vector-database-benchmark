from __future__ import annotations
from typing import List, Sequence, Tuple
from pyinstrument.frame import DUMMY_ROOT_FRAME_IDENTIFIER, SELF_TIME_FRAME_IDENTIFIER, Frame
from pyinstrument.frame_info import frame_info_get_identifier
from pyinstrument.typing import LiteralStr, assert_never
FrameRecordType = Tuple[List[str], float]

class IdentifierDoesntMatchException(ValueError):
    pass

def build_frame_tree(frame_records: Sequence[FrameRecordType]) -> Frame | None:
    if False:
        i = 10
        return i + 15
    if len(frame_records) == 0:
        return None
    root_frame = Frame(identifier_or_frame_info=DUMMY_ROOT_FRAME_IDENTIFIER)
    frame_stack: list[Frame] = [root_frame]
    for (frame_info_stack, time) in frame_records:
        stack_depth = 0
        root_frame.record_time_from_frame_info(DUMMY_ROOT_FRAME_IDENTIFIER, time)
        for (stack_depth, frame_info) in enumerate(frame_info_stack, start=1):
            frame_identifier = frame_info_get_identifier(frame_info)
            try:
                frame = frame_stack[stack_depth]
                if frame.identifier != frame_identifier:
                    del frame_stack[stack_depth:]
                    raise IdentifierDoesntMatchException()
            except (IndexError, IdentifierDoesntMatchException):
                parent = frame_stack[stack_depth - 1]
                frame = Frame(identifier_or_frame_info=frame_info)
                parent.add_child(frame)
                assert len(frame_stack) == stack_depth
                frame_stack.append(frame)
            frame.record_time_from_frame_info(frame_info=frame_info, time=time)
        del frame_stack[stack_depth + 1:]
        final_frame = frame_stack[-1]
        if not final_frame.is_synthetic_leaf:
            final_frame.add_child(Frame(identifier_or_frame_info=SELF_TIME_FRAME_IDENTIFIER, time=time))
    if len(root_frame.children) == 1:
        root_frame = root_frame.children[0]
        root_frame.remove_from_parent()
    return root_frame

def delete_frame_from_tree(frame: Frame, replace_with: LiteralStr['children', 'self_time', 'nothing']):
    if False:
        while True:
            i = 10
    '\n    Delete a frame from the tree.\n\n    :param frame: the frame to delete\n    :param replace_with: what to replace the frame with - `children` replaces\n        the frame with its children, `self_time` replaces the frame with a\n        self-time frame, and `nothing` deletes the frame, absorbing the time\n        into the parent.\n    '
    parent = frame.parent
    if parent is None:
        raise ValueError('Cannot delete the root frame')
    if replace_with == 'children':
        parent.add_children(frame.children, after=frame)
    elif replace_with == 'self_time':
        parent.add_child(Frame(identifier_or_frame_info=SELF_TIME_FRAME_IDENTIFIER, time=frame.time), after=frame)
    elif replace_with == 'nothing':
        parent.absorbed_time += frame.time
    else:
        assert_never(replace_with)
    parent.absorbed_time += frame.absorbed_time
    frame.remove_from_parent()
    remove_frame_from_groups(frame, recursive=True)

def combine_frames(frame: Frame, into: Frame):
    if False:
        i = 10
        return i + 15
    '\n    Combine two frames into one. The frames must have the same parent.\n\n    :param frame: the frame to remove\n    :param into: the frame to combine into\n    '
    assert frame.parent is into.parent
    into.absorbed_time += frame.absorbed_time
    into.time += frame.time
    for (attribute, time) in frame.attributes.items():
        try:
            into.attributes[attribute] += time
        except KeyError:
            into.attributes[attribute] = time
    into.add_children(frame.children)
    frame.remove_from_parent()
    remove_frame_from_groups(frame, recursive=False)

def remove_frame_from_groups(frame: Frame, recursive: bool):
    if False:
        return 10
    "\n    Removes frame from any groups that it is a member of. Should be used when\n    removing a frame from a tree, so groups don't keep references to removed\n    frames.\n    "
    if recursive and frame.children:
        for child in frame.children:
            remove_frame_from_groups(child, recursive=True)
    if frame.group:
        group = frame.group
        group.remove_frame(frame)
        if len(group.frames) == 1:
            group.remove_frame(group.frames[0])