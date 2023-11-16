scene	cue	follow	name	label	tracks	behavior	timestop	timer	soundintro	soundeval	soundround	soundtrack	soundsynth	colorset	
preparation	0.10	0.15	register	0.10 register	Reset	default	1	Reinit							
preparation	0.15	0.20	walkin	0.15 walkin			1								
preparation	0.20	0.30	field	0.20 field			1								
preparation	0.30	0.40	boundaries	0.30 boundaries			1								
preparation	0.40	0.50	entrance	0.40 entrance			1						1 2000		
preparation	0.50	1.10	black	0.50 black			1								
prolog	1.10	1.20	beginning	1.10 beginning			1								
prolog	1.20	1.30	prolog	1.20 prolog			1		prolog						
prolog	1.30	1.35	boot	1.30 boot			1				boot				
prolog	1.35	1.40	platforms pre	1.35 platform pre			1								
prolog	1.40	2.10	platforms	1.40 platforms			1						1 2000		
polls	2.10	2.20	experimentstart	2.10 experimentstart			1		poll				0 5000		
polls	2.20	3.04	poll	2.20 poll			1						1 2000		
association	3.00	3.04	assoc load	3.00 assoc load	Initround	default	1	Reinit							
association	3.04	3.05	association_score	3.04 assoc splash	Initround		1		scores						
association	3.05	3.10.0	association_intro	3.05 assoc intro	Initround		1		association						
association	3.10	3.10	association_ruleset	3.10 assoc ruleset vorlage	Startround	default	0	Goround				01 1 2000	1 2000		
association	3.10.0	3.10.1	association_ruleset1	3.10 assoc ruleset1	Startround	default	0	Goround				01 1 2000	1 2000		
association	3.10.1	3.10.2	association_repeat	3.10 assoc repeat			0		associationrepeat			01 -1 2000	1 2000		
association	3.10.2	3.16	association_ruleset2	3.10 assoc ruleset2			0					01 -1 2000	1 2000		
association	3.16	4.04	association_countdown	3.16 assoc countdown		default	0	Gocd			countdown	01 -1 2000	1 2000		
association	3.30	3.40	association_lab	3.30 assoc lab	Stopround		1	Reinit		1					
association	3.40	3.41	association_selection1	3.40 assoc selection 1			1								
association	3.41	3.42	association_selection1	3.41 assoc selection 2			1								
association	3.42	3.43	association_selection1	3.42 assoc selection 3			1								
association	3.43	3.44	association_profile1	3.43 assoc profile 1			1								
association	3.44	3.45	association_profile2	3.44 assoc profile 2			1								
association	3.45	4.04	association_profile3	3.45 assoc profile 3			1								
movement	4.00	4.04	movement_load	4.00 mov load	Initround	default	1	Reinit							
movement	4.04	4.05	movement_splashscreen	4.04 mov splash			1	Reinit							
movement	4.05	4.10.0	movement_intro	4.05 mov intro			1	Reinit	movement						
movement	4.10	4.10	movement_ruleset	4.10 mov ruleset vorlage	Startround	default	0	Goround				02 1 2000			
movement	4.10.0	4.10.1	movement_ruleset1	4.10 mov ruleset1	Startround	default	0	Goround				02 1 2000			
movement	4.10.1	4.10.2	movement_repeat	4.10 mov rule repeat			0		movementrepeat			02 -1 2000			
movement	4.10.2	4.12	movement_ruleset2	4.10 mov ruleset2			0					02 -1 2000			
movement	4.12	4.14	movement_plateau1	4.12 mov plateau 1			0				plateau	02 2 0			
movement	4.14	4.16	movement_plateau2	4.14 mov plateau 2			0				plateau	02 3 0			
movement	4.16	4.30	movement_countdown	4.16 mov countdown		default	0	Gocd			countdown	02 -1 2000			
movement	4.20	4.30	movement_conform	4.20 mov conform		conform	1				conformbehavior				
movement	4.25	4.30	movement_rebel	4.25 mov rebel		rebel	1				rebelbehavior				
movement	4.30	4.40	movement_lab	4.30 mov lab	Stopround		1			1					
movement	4.40	4.43	movement_profilegroup	4.40 mov profile group			1								
movement	4.43	4.44	movement_profileworst	4.43 mov profile worst			1								
movement	4.44	5.00	movement_profilebest	4.44 mov profile best			1								
movement	4.45	5.00	movement_profilemiddle	4.45 mov profile middle			1								
distance	5.00	5.04	distance_load	5.00 dist load	Initround	default	1	Reinit							
distance	5.04	5.05	distance_splash	5.04 dist splash	Initround	default	1	Reinit							
distance	5.05	5.10.0	distance_intro	5.05 dist intro	Initround	default	1	Reinit	distance						
distance	5.10	5.10	distance_ruleset	5.10 distance ruleset vorlage	Startround		0	Goround				03 1 2000			
distance	5.10.0	5.10.1	distance_ruleset1	5.10 distance ruleset1	Startround		0	Goround				03 1 2000			
distance	5.10.1	5.10.2	distance_repeat	5.10 distance repeat			0		distancerepeat			03 -1 2000			
distance	5.10.2	5.12	distance_ruleset2	5.10 distance ruleset2			0					03 -1 2000			
distance	5.12	5.14	distance_plateau1	5.12 distance plateau 1			0				plateau	03 2 0			
distance	5.14	5.16	distance_plateau2	5.14 distance plateau 2			0				plateau	03 3 0			
distance	5.16	5.30	distance_countdown	5.16 distance countdown		default	0	Gocd			countdown	03 -1 2000			
distance	5.20	5.30	distance_conform	5.20 distance conform		conform	1				conformbehavior				
distance	5.25	5.30	distance_rebel	5.25 distance rebel		rebel	1				rebelbehavior				
distance	5.30	5.40	distance_lab	5.30 distance lab	Stopround		1			1					
distance	5.40	5.43	distance_profilegroup	5.40 distance profile group			1								
distance	5.43	5.44	distance_profilebest	5.43 distance profile best			1								
distance	5.44	6.00	distance_profileworst	5.44 distance profile worst			1								
distance	5.45	6.00	distance_profilemiddle	5.45 distance profile middle			1								
prediction	6.00	6.04	prediction_load	6.00 prediction load	Initround	default	1	Reinit							
prediction	6.04	6.05	prediction_splash	6.04 prediction splash	Initround	default	1	Reinit							
prediction	6.05	6.10.0	prediction_intro	6.05 prediction intro	Initround	default	1	Reinit	prediction						
prediction	6.10	6.10	prediction_ruleset	6.10 prediction ruleset vorlage	Startround		0	Goround				04 1 2000			
prediction	6.10.0	6.10.1	prediction_ruleset1	6.10 prediction ruleset1	Startround		0	Goround				04 1 2000			
prediction	6.10.1	6.10.2	prediction_repeat	6.10 prediction repeat			0		predictionrepeat			04 -1 2000			
prediction	6.10.2	6.16	prediction_ruleset2	6.10 prediction ruleset2			0					04 -1 2000			
prediction	6.16	6.30	prediction_countdown	6.16 prediction countdown		default	0	Gocd			countdown	04 -1 2000			
prediction	6.20	6.30	prediction_conform	6.20 prediction_conform		conform	0				conformbehavior				
prediction	6.25	6.30	prediction_rebel	6.25 prediction_rebel		rebel	1				rebelbehavior				
prediction	6.30	6.43	prediction_lab	6.30 prediction_lab	Stopround		1			1					
prediction	6.40	6.43	prediction_groupprofile	6.40 prediction_groupprofile			1								
prediction	6.43	6.44	prediction_profilebest	6.43 prediction_profilebest			1								
prediction	6.44	7.00	prediction_profilemiddle	6.44 prediction_profilemiddle			1								
custom	7.00	7.04	custom_load	7.00 custom_load			1	Reinit							
custom	7.04	7.05	custom_splash	7.04 custom_splash			1	Reinit							
custom	7.05	7.10	custom_intro	7.05 custom_intro			1	Reinit	custom						
custom	7.10	7.11	custom_question_dist	7.10 custom_question_dist			1						1 2000		
custom	7.11	7.20	custom_poll_dist	7.11 custom_poll_dist			1						1 0		
custom	7.20	7.21	custom_question_assoc	7.20 custom_question_assoc			1						1 0		
custom	7.21	7.30	custom_poll_assoc	7.21 custom_poll_assoc			1						1 0		
custom	7.30	7.31	custom_question_mov	7.30 custom_question_mov			1						1 0		
custom	7.31	7.40	custom_poll_mov	7.31 custom_poll_mov			1						1 0		
custom	7.40	7.41	custom_question_pred	7.40 custom_question_pred			1						1 0		
custom	7.41	7.50	custom_poll_pred	7.41 custom_poll_pred			1						1 0		
custom	7.50	7.51	custom_poll_cyber	7.50 custom_question_cyber			1						1 0		
custom	7.51	7.52	custom_question_cyber	7.51 custom_poll_cyber			1						1 0		
custom	7.52	7.70	custom_black	7.52 custom_black			1						0 5000		
custom	7.64	7.65	custom_splash	custom splash	Initround		1	Reinit							
custom	7.65	7.66	custom_intro	custom intro	Initround		1		playcustom						
custom	7.66	7.70	custom_round	custom round	Startround		1						1 2000		
custom	7.70	8.10	custom_reset	7.70 custom_reset	Stopround		1				reset				
epilog	8.10	8.20	epilog_black	8.10 epilog_talk			1		epilog						
epilog	8.20	8.20	epilog_black	8.20 epilog_black			1								
joker	23	23	joker	joker			1				joker				
reset	123	123	reset	reset			1	Reset			reset				
