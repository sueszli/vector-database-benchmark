import datetime
import math
from pony import orm
from pony.orm import db_session
from tribler.core.components.metadata_store.db.orm_bindings.channel_node import LEGACY_ENTRY

def define_binding(db):
    if False:
        return 10

    class Vsids(db.Entity):
        """
        This ORM class is used to hold persistent information for the state of VSIDS scoring system.
        ACHTUNG! At all times there should be no more than one row/entity of this class. A single entity is
        enough to keep the information for the whole GigaChannels.
        In a sense, *this is a singleton*.
        """
        rowid = orm.PrimaryKey(int)
        bump_amount = orm.Required(float)
        total_activity = orm.Required(float)
        last_bump = orm.Required(datetime.datetime)
        rescale_threshold = orm.Optional(float, default=10.0 ** 100)
        exp_period = orm.Optional(float, default=24.0 * 60 * 60 * 3)
        max_val = orm.Optional(float, default=1.0)

        @db_session
        def rescale(self, norm):
            if False:
                i = 10
                return i + 15
            for channel in db.ChannelMetadata.select(lambda g: g.status != LEGACY_ENTRY):
                channel.votes /= norm
            for vote in db.ChannelVote.select():
                vote.last_amount /= norm
            self.max_val /= norm
            self.total_activity /= norm
            self.bump_amount /= norm
            db.ChannelMetadata.votes_scaling = self.max_val

        @db_session
        def normalize(self):
            if False:
                print('Hello World!')
            self.total_activity = self.total_activity or orm.sum((g.votes for g in db.ChannelMetadata))
            channel_count = orm.count(db.ChannelMetadata.select(lambda g: g.status != LEGACY_ENTRY))
            if not channel_count:
                return
            if self.total_activity > 0.0:
                self.rescale(self.total_activity / channel_count)
                self.bump_amount = 1.0

        @db_session
        def bump_channel(self, channel, vote):
            if False:
                print('Hello World!')
            now = datetime.datetime.utcnow()
            channel.votes -= vote.last_amount
            self.total_activity -= vote.last_amount
            self.bump_amount *= math.exp((now - self.last_bump).total_seconds() / self.exp_period)
            self.last_bump = now
            vote.last_amount = self.bump_amount
            channel.votes += self.bump_amount
            self.total_activity += self.bump_amount
            if channel.votes > self.max_val:
                self.max_val = channel.votes
            db.ChannelMetadata.votes_scaling = self.max_val
            if self.bump_amount > self.rescale_threshold:
                self.rescale(self.bump_amount)

        @classmethod
        @db_session
        def create_default_vsids(cls):
            if False:
                return 10
            return cls(rowid=0, bump_amount=1.0, total_activity=orm.sum((g.votes for g in db.ChannelMetadata)) or 0.0, last_bump=datetime.datetime.utcnow())
    return Vsids