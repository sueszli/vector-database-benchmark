from TDStoreTools import StorageManager
import TDFunctions as TDF

class Utils:
	"""
	Control Utils description
	"""
	def __init__(self, ownerComp):
		self.ownerComp = ownerComp
		self.pars = ownerComp.par
		self.Loaded = ownerComp.op('./loaded_cue')
		self.Sceneloader = op.Scene
		self.JokerState = 0

	def JokerButton(self):
		if self.JokerState == 0:
			self.JokerState = 1
			ctable = op('cuetable')
			visfreeze = op.Outputfreeze
			visfreeze.par.Value0 = 1
			jokercue = ctable['joker','name'].row
			self.GoTo(jokercue)

	def Pause(self):
		self.pars.Timestop.val = 1

	def Unpause(self):
		self.pars.Timestop.val = 0

	def Go(self):
		op.Scene.Load()
		op.Scene.Go()
		self.GoScene()
		self.GoSound()
		self.GoLight()
		self.GoTracks()
		self.GoTimer()
		self.GoBehavior()

	def GoTo(self, cueIndex):
		sop = op.Scene
		target = tdu.clamp(int(cueIndex), 1, (sop.par.Size.eval() - 1))
		sop.par.Index.val = target
		self.Go()

	def Arm(self):
		self.Sceneloader.par.Index.val = int(self.pars.Preloadindex.eval())

	def GoRelative(self, step):
		sop = op.Scene.par
		curi = sop.Index.eval()
		nexti = tdu.clamp((curi + step), 1, (sop.Size - 1))
		sop.Index = nexti
		self.Go()

	def GoNext(self):
		target = int(self.pars.Followindex.eval())
		self.GoTo(target)

	def GoBack(self):
		target = int(self.pars.Previousindex.eval())
		self.GoTo(target)

	def GoTable(self, ref):
		table = op(ref)
		for i in range(1, table.numRows):
			name = table[i,'parameter']
			value = table[i,'value']
			path = table[i,'path']
			target = op(path).par[name]
			try:
				target.val = value.val
			except AttributeError:
				debug(f"Error while trying to set parameter via cuetable:\nName: {name}\nPath: {path}\nValue: {value}")
				pass
			pass

	def GoScene(self):
		self.pars.Timestop = int(self.Loaded[1,'stop'].val)
		for fop in ops('scene_*'):
			self.GoTable(fop)

	def GoBehavior(self):
		behavior = self.Loaded[1,'behavior'].val
		if not behavior == '':
			if behavior == 'default':
				op.Group.par.Roundfinishearly.val = 0
				op.Group.par.Roundfinishcriticalmass.val = 0
			elif behavior == 'conform':
				op.Group.par.Roundfinishearly.val = 1
				op.Group.par.Roundfinishcriticalmass.val = 0
			elif behavior == 'rebel':
				op.Group.par.Roundfinishearly.val = 1
				op.Group.par.Roundfinishcriticalmass.val = 1

	def GoTracks(self):
		cmd = str(self.Loaded[1,'tracks'].val)
		try:
			op.Tracker.par[cmd].pulse()
		except:
			pass

	def GoTimer(self):
		cmd = str(self.Loaded[1,'timer'].val)
		try:
			op.Roundtimer.par[cmd].pulse()
		except:
			pass

	def GoGraphics(self):
		rendercue = self.Loaded[1,'rendering'].val
		colorcue = self.Loaded[1,'colorset'].val
		# debug(rendercue)
		op.Rendercl.Recall_Cue(rendercue)
		#op.Colorcl.Recall_Cue(colorcue)

	def GoSound(self):
		scene = self.Loaded[1,'scene']
		soundIntro =  str(self.Loaded[1,'soundintro'].val)
		soundRound = str(self.Loaded[1,'soundround'].val)
		soundSynth = (self.Loaded[1,'synth'].val or False)
		soundTrack = (self.Loaded[1,'soundtrack'].val or False)
		op.Sound.SendScene(scene)
		if soundSynth:
			args = soundSynth.split(' ')
			op.Sound.SendSynthtoggle(trigger = args[0], fademillis = args[1])
		else:
			op.Sound.SendSynthtoggle(soundSynth)
		if soundTrack:
			args = soundTrack.split(' ')			
			op.Sound.SendSoundtrack(subtype = args[0], trigger = int(args[1]), fademillis = args[2])
		else:
			op.Sound.SendSoundtrack()
		if (soundIntro == '') and (soundRound == ''):
			op.Sound.SendAblVoiceVol(trigger = 1, fademillis = 1000)
			op.Sound.SendRound('joker', [0])
		else:			
			op.Sound.SendAblVoiceVol(trigger = 0, fademillis = 1000)				
			if not soundIntro == '':
				op.Sound.SendIntro(soundIntro)
			if soundRound != 'joker':
				op.Sound.SendRound('joker', [0])
			if not soundRound == '':
				op.Sound.SendRound(soundRound)

	def GoLight(self):
		cue = str(self.Loaded[1,'cue'].val)
		event = str(self.Loaded[1,'lights'].val)
		op.Guide.SendCue(cue)
		if event == '/joker':
			op.Guide.SendJoker(1)
		else:
			op.Guide.SendJoker(0)
			if event != '':
				op.Guide.SendEvent(event)