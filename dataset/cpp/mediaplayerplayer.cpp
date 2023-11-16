#include "mediaplayerplayer.hpp"
#include "service.hpp"

#ifdef USE_DBUS

mp::MediaPlayerPlayer::MediaPlayerPlayer(lib::spt::api &spotify, QObject *parent)
	: QDBusAbstractAdaptor(parent),
	dBus(QDBusConnection::sessionBus()),
	spotify(spotify)
{
	callback = [](const std::string &/*result*/)
	{
		// We trust that the error message has already been logged somewhere
	};
}

void mp::MediaPlayerPlayer::Next() const
{
	spotify.next(callback);
}

void mp::MediaPlayerPlayer::Pause() const
{
	spotify.pause(callback);
}

void mp::MediaPlayerPlayer::Play() const
{
	spotify.resume(callback);
}

void mp::MediaPlayerPlayer::PlayPause() const
{
	if (currentPlayback().is_playing)
	{
		spotify.pause(callback);
	}
	else
	{
		spotify.resume(callback);
	}
}

void mp::MediaPlayerPlayer::Previous() const
{
	spotify.previous(callback);
}

void mp::MediaPlayerPlayer::Seek(qint64 offset) const
{
	const auto position = (currentPlayback().progress_ms + offset) * msInUs;
	spotify.seek(static_cast<int>(position), callback);
}

void mp::MediaPlayerPlayer::SetPosition(const QDBusObjectPath &/*trackId*/, qint64 position) const
{
	spotify.seek(static_cast<int>(position * msInUs), callback);
}

void mp::MediaPlayerPlayer::Stop() const
{
	spotify.pause(callback);
}

auto mp::MediaPlayerPlayer::canControl() const -> bool
{
	return true;
}

auto mp::MediaPlayerPlayer::metadata() const -> QMap<QString, QVariant>
{
	return Json::toVariantMap(currentPlayback().metadata());
}

auto mp::MediaPlayerPlayer::getVolume() const -> double
{
	return currentPlayback().volume() / 100.0;
}

void mp::MediaPlayerPlayer::setVolume(double value) const
{
	spotify.set_volume((int) (value * 100), callback);
}

auto mp::MediaPlayerPlayer::position() const -> qint64
{
	return static_cast<qint64>(currentPlayback().progress_ms * 1000);
}

auto mp::MediaPlayerPlayer::playbackStatus() const -> QString
{
	return currentPlayback().is_playing ? "Playing" : "Paused";
}

void mp::MediaPlayerPlayer::OpenUri(const QString &uri) const
{
	if (uri.startsWith(QStringLiteral("spotify:track:")))
	{
		spotify.play_tracks(0, {uri.toStdString()}, callback);
		return;
	}

	if (uri.startsWith(QStringLiteral("https://open.spotify.com/track/")))
	{
		const auto urlUri = lib::spt::url_to_uri(uri.toStdString());
		spotify.play_tracks(0, {urlUri}, callback);
		return;
	}

	lib::log::warn("\"{}\" is not a valid Spotify track URL/URI", uri.toStdString());
}

auto mp::MediaPlayerPlayer::playbackRate() const -> double
{
	return 1.0;
}

void mp::MediaPlayerPlayer::setPlaybackRate(double /*value*/) const
{
	lib::log::warn("Changing playback rate is not supported");
}

auto mp::MediaPlayerPlayer::shuffle() const -> bool
{
	return currentPlayback().shuffle;
}

void mp::MediaPlayerPlayer::setShuffle(bool value) const
{
	spotify.set_shuffle(value, callback);
}

auto mp::MediaPlayerPlayer::getLoopStatus() const -> QString
{
	switch (currentPlayback().repeat)
	{
		case lib::repeat_state::track:
			return QStringLiteral("Track");

		case lib::repeat_state::context:
			return QStringLiteral("Playlist");

		case lib::repeat_state::off:
			return QStringLiteral("None");

		default:
			return {};
	}
}

void mp::MediaPlayerPlayer::setLoopStatus(const QString &loopStatus)
{
	lib::repeat_state repeatState;

	if (loopStatus == QStringLiteral("Track"))
	{
		repeatState = lib::repeat_state::track;
	}
	else if (loopStatus == QStringLiteral("Playlist"))
	{
		repeatState = lib::repeat_state::context;
	}
	else if (loopStatus == QStringLiteral("None"))
	{
		repeatState = lib::repeat_state::off;
	}
	else
	{
		return;
	}

	spotify.set_repeat(repeatState, callback);
}

void mp::MediaPlayerPlayer::emitMetadataChange() const
{
	QVariantMap properties;
	properties["Metadata"] = Json::toVariantMap(currentPlayback().metadata());
	Service::signalPropertiesChange(this, properties);
}

void mp::MediaPlayerPlayer::currentSourceChanged() const
{
	QVariantMap properties;
	properties["Metadata"] = Json::toVariantMap(currentPlayback().metadata());
	properties["PlaybackStatus"] = currentPlayback().is_playing ? "Playing" : "Paused";
	//properties["CanSeek"] = true;
	Service::signalPropertiesChange(this, properties);
}

void mp::MediaPlayerPlayer::stateUpdated() const
{
	QVariantMap properties;
	properties["PlaybackStatus"] = currentPlayback().is_playing
		? QStringLiteral("Paused")
		: QStringLiteral("Playing");

	Service::signalPropertiesChange(this, properties);
}

void mp::MediaPlayerPlayer::totalTimeChanged() const
{
	QVariantMap properties;
	properties["Metadata"] = Json::toVariantMap(currentPlayback().metadata());
	Service::signalPropertiesChange(this, properties);
}

void mp::MediaPlayerPlayer::seekableChanged(bool seekable) const
{
	QVariantMap properties;
	properties["CanSeek"] = seekable;
	Service::signalPropertiesChange(this, properties);
}

void mp::MediaPlayerPlayer::volumeChanged() const
{
	QVariantMap properties;
	properties["Volume"] = currentPlayback().volume() / 100.0;
	Service::signalPropertiesChange(this, properties);
}

void mp::MediaPlayerPlayer::seeked(qint64 newPos)
{
	emit Seeked(newPos * msInUs);
}

void mp::MediaPlayerPlayer::setCurrentPlayback(const lib::spt::playback &/*playback*/)
{
}

auto mp::MediaPlayerPlayer::currentPlayback() const -> lib::spt::playback
{
	return ((Service *) parent())->currentPlayback();
}

#endif
