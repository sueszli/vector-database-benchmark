"""
TouchDesigner Utilities for SystemFailed Sound Component, 
mostly convenience wrappers for common OSC sound directives.
"""

from TDStoreTools import StorageManager
import TDFunctions as TDF

class Utils:
	"""
	Utils for the SystemFailed Sound comp
	""" 
	def __init__(self, ownerComp):
		self.ownerComp = ownerComp
		self.pars = ownerComp.par
		maxmsp1 = op('sender_maxmsp')
		maxmsp2 = op('sender_debug_maxmsp')
		ableton1 = op('sender_ableton')
		ableton2 = op('sender_debug_ableton')
		synth1 = op('sender_synth')
		synth2 = op('sender_synth_debug')
		zap1 = op('sender_zap')
		zap2 = op('sender_zap_debug')
		magicq1 = op('sender_magicq')
		magicq2 = op('sender_magicq_debug')

		self.synthSet = [i for i in range(1,51)]
		self.synthIndex = 0

		self.maxmspSenders = [maxmsp1, maxmsp2]
		self.abletonSenders = [ableton1, ableton2]
		self.synthSenders = [synth1, synth2]
		self.zapSenders = [zap1, zap2]
		self.magicqSenders = [magicq1, magicq2]
		self.zaps = dict()
		self.zUnassigned = set(range(7))
		self.strobes = dict()
		self.sUnassigned = set(range(4))

	def SendMaxmsp(self, message, args):
		for s in self.maxmspSenders:
			# debug(f'{self.ownerComp} sending osc on {s}:\n {message}, {args}')
			s.sendOSC(message, args, asBundle=False, useNonStandardTypes=True)
		return

	def SendZap(self, message, args):
		for s in self.zapSenders:
			# debug(f'{self.ownerComp} sending osc on {s}:\n {message}, {args}')
			s.sendOSC(message, args, asBundle=False, useNonStandardTypes=True)
		return

	def SendAbleton(self, message, args):
		for s in self.abletonSenders:
			# debug(f'{self.ownerComp} sending osc on {s}:\n {message}, {args}')
			s.sendOSC(message, args, asBundle=False, useNonStandardTypes=True)
		return

	def SendMagicq(self, message, args):
		msg = f'/round/{message}'
		for s in self.magicqSenders:
			# debug(f'{self.ownerComp} sending osc on {s}:\n {message}, {args}')
			s.sendOSC(msg[0:15], args, asBundle=False, useNonStandardTypes=True)
		return

	def SendSynth(self, message, args):
		for s in self.synthSenders:
			# debug(f'{self.ownerComp} sending osc on {s}:\n {message}, {args}')
			s.sendOSC(message, args, asBundle=False, useNonStandardTypes=True)
		return

	def SendScene(self, name):
		msg = f'/scene'
		args = [str(name).lower()]
		self.SendMaxmsp(msg, args)
		return

	def SendSetting(self, subtype):
		msg = f'/setting/{subtype}'
		self.SendMaxmsp(msg, [int(1)])
		return

	def SendIntro(self, name):
		if self.pars.Intro.eval():
			msg = f'/intro/{name.lower()}'
			args = [int(1)]
			self.SendAbleton(msg, args)
		return

	def SendVoice(self, subtype, arguments):
		msg = f'/voice/{subtype}'
		args = [int(a) for a in arguments]
		self.SendMaxmsp(msg, args)
		return

	def SendRound(self, subtype, arguments = [1]):
		if self.pars.Round.eval():
			msg = f'/round/{subtype}'
			args = [int(a) for a in arguments]
			self.SendAbleton(msg, args)
		return

	def SendAblVoiceVol(self, trigger = 1, fademillis = 1000):
		msg = f'/aivoice/vol'
		args = [int(trigger), int(fademillis)]
		self.SendAbleton(msg, args)

	def SendFreeze(self, subtype, trackid):
		if self.pars.Freeze.eval():
			newType = f'freeze/{subtype}'
			self.SendVoice(newType, [trackid])
			# debug(f"freeze: {newType}, {trackid}")
		return

	def SendBenched(self, subtype, trackid):
		if self.pars.Bench.eval():
			newType = f'benched/{subtype}'
			self.SendVoice(newType, [trackid])
			# debug(f"bench: {newType}, {trackid}")
		return

	def SendEvaluationStart(self, trigger = 1):
		self.SendRound('evaluation/start', [trigger])
		return

	def SendEvaluationRank(self, subtype, rank):
		newType = f'evaluation/{subtype}'
		hs = op('highscore_set')
		if subtype == 'high':
			ref = rank
		else:
			ref = hs.numSamples - rank
		trackid = hs['Trackid'][rank]
		score = hs['Newhighscore'][rank]
		self.SendVoice(newType, [trackid, score, rank])
		return

	def SendConformEnd(self, trigger = 1):
		self.SendRound('conformbehavior',[trigger])
		return

	def SendRebelEnd(self, trigger = 1):
		self.SendRound('rebelbehavior',[trigger])
		return

	def SendCountdown(self, trigger = 1):
		self.SendRound('countdown', [int(trigger)])
		return

	def SendSoundLocalized(self, subtype, slot = 0, trigger = 1, posx = 0, posy = 0):
		self.SendZap(f'/sound/{subtype}', [int(slot),int(trigger), float(posx), float(posy)])
		return

	def SendSynthSingle(self, pitch = 1, level = 0, posx = 0, posy = 0):
		if self.pars['Synth']:
			self.SendSynth(f'/synth', [int(pitch), float(level), float(posx), float(posy)])
		return

	def SendSynthCycle(self):
		# self.synthSet = [i for i in range(1,51)]
		synth = op('synth_set_dat')
		for i in range(5): 
			self.synthIndex = ((self.synthIndex + 1) % 50)
			pitch = int(synth[self.synthIndex+1,'Trackid'].val)
			# args.append(pitch)
			level = float(synth[self.synthIndex+1,'Level'].val or 0)
			# args.append(level)
			posx = float(synth[self.synthIndex+1,'Positionx'].val or 0)
			# args.append(posx)
			posy = float(synth[self.synthIndex+1,'Positiony'].val or 0)
			# args.append(posy)
			op.Sound.SendSynthSingle(pitch, level, posx, posy)
			# debug(self.synthIndex+1)
		# self.SendSynth(f'/synth', args)
		return

	def SendSynthBundle(self, args):
		self.SendSynth(f'/synth', args)
		return

	def SendSynthtoggle(self, trigger = 1, fademillis = 3000):
		self.SendMaxmsp(f'/synthtoggle', [int(trigger), int(fademillis)])
		return

	def SendSoundtrack(self, subtype = '0', trigger = 1, fademillis = 3000):
		if subtype == '0':
			trigger = 0
		if trigger == -1:
			pass
		else:
			self.SendAbleton(f'/soundtrack/{subtype}', [int(trigger), int(fademillis)])
		return

	# list tracks: [trackid, px, py]
	def SendZaps(self, tracks):
		tmp = dict()
		deletes = set()
		zaps = self.zaps
		if len(tracks) > 1:
			for track in tracks:
				tid = track[0]
				tx = track[1]
				ty = track[2]
				tmp[tid] = (tid,tx,ty)
		else:
			for offid in range(7):
				self.SendSoundLocalized(subtype='zap', slot=offid, trigger=0, posx=0, posy=0)
		for tid in tmp.keys():
			# vals = tmp[tid]
			if tid in zaps.keys():
				# RETRIGGER
				slotid = zaps[tid][0]
				zaps[tid] = (slotid, tmp[tid][0], tmp[tid][1], tmp[tid][2])
				# debug(f'zap retrigger {zaps[tid]}')
				self.SendSoundLocalized(subtype='zap', slot=slotid, trigger=-1, posx=zaps[tid][2], posy=zaps[tid][3])
			else: 
				if len(self.zUnassigned) == 0:
					pass
				# TRIGGER
				slotid = self.zUnassigned.pop()
				zaps[tid] = (slotid, tmp[tid][0], tmp[tid][1], tmp[tid][2])
				# debug(f'zap trigger {zaps[tid]}')
				self.SendSoundLocalized(subtype='zap', slot=slotid, trigger=1, posx=zaps[tid][2], posy=zaps[tid][3])
		for tid in zaps.keys():
			if not (tid in tmp.keys()):
				deletes.add(tid)
		for tid in deletes:
			# OFF
			self.SendSoundLocalized(subtype='zap', slot=zaps[tid][0], trigger=0, posx=zaps[tid][2], posy=zaps[tid][3])
			self.zUnassigned.add(zaps[tid][0])
			try:
				zaps.pop(tid)
			except KeyError:
				pass

	def SendStrobes(self, tracks):
		tmp = dict()
		deletes = set()
		strobes = self.strobes
		for track in tracks:
			tid = track[0]
			tx = track[1]
			ty = track[2]
			tmp[tid] = (tid,tx,ty)
		for tid in tmp.keys():
			# vals = tmp[tid]
			if tid in strobes.keys():
				# RETRIGGER
				slotid = strobes[tid][0]
				strobes[tid] = (slotid, tmp[tid][0], tmp[tid][1], tmp[tid][2])
				self.SendSoundLocalized(subtype='strobe', slot=slotid, trigger=-1, posx=strobes[tid][2], posy=strobes[tid][3])
			else: 
				if len(self.sUnassigned) == 0:
					pass
				# TRIGGER
				slotid = self.sUnassigned.pop()
				strobes[tid] = (slotid, tmp[tid][0], tmp[tid][1], tmp[tid][2])
				self.SendSoundLocalized(subtype='strobe', slot=slotid, trigger=1, posx=strobes[tid][2], posy=strobes[tid][3])
		for tid in strobes.keys():
			if not (tid in tmp.keys()):
				deletes.add(tid)
		for tid in deletes:
			# OFF
			self.SendSoundLocalized(subtype='strobe', slot=strobes[tid][0], trigger=0, posx=strobes[tid][2], posy=strobes[tid][3])
			self.sUnassigned.add(strobes[tid][0])
			strobes.pop(tid)