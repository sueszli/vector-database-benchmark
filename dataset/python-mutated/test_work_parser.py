import unittest
from wechatpy.work import events, parse_message

class ParseMessageTestCase(unittest.TestCase):

    def test_subscribe_event(self):
        if False:
            print('Hello World!')
        xml = '\n        <xml>\n            <ToUserName><![CDATA[toUser]]></ToUserName>\n            <FromUserName><![CDATA[UserID]]></FromUserName>\n            <CreateTime>1348831860</CreateTime>\n            <MsgType><![CDATA[event]]></MsgType>\n            <Event><![CDATA[subscribe]]></Event>\n            <AgentID>1</AgentID>\n        </xml>\n        '
        event = parse_message(xml)
        self.assertIsInstance(event, events.SubscribeEvent)
        self.assertEqual(1, event.agent)

    def test_parse_text_message(self):
        if False:
            for i in range(10):
                print('nop')
        xml = '<xml>\n        <ToUserName><![CDATA[toUser]]></ToUserName>\n        <FromUserName><![CDATA[fromUser]]></FromUserName>\n         <CreateTime>1348831860</CreateTime>\n        <MsgType><![CDATA[text]]></MsgType>\n        <Content><![CDATA[this is a test]]></Content>\n        <MsgId>1234567890123456</MsgId>\n        <AgentID>1</AgentID>\n        </xml>'
        msg = parse_message(xml)
        self.assertEqual('text', msg.type)
        self.assertEqual(1, msg.agent)

    def test_parse_image_message(self):
        if False:
            i = 10
            return i + 15
        xml = '<xml>\n        <ToUserName><![CDATA[toUser]]></ToUserName>\n        <FromUserName><![CDATA[fromUser]]></FromUserName>\n        <CreateTime>1348831860</CreateTime>\n        <MsgType><![CDATA[image]]></MsgType>\n        <PicUrl><![CDATA[this is a url]]></PicUrl>\n        <MediaId><![CDATA[media_id]]></MediaId>\n        <MsgId>1234567890123456</MsgId>\n        <AgentID>1</AgentID>\n        </xml>'
        msg = parse_message(xml)
        self.assertEqual('image', msg.type)

    def test_parse_voice_message(self):
        if False:
            for i in range(10):
                print('nop')
        xml = '<xml>\n        <ToUserName><![CDATA[toUser]]></ToUserName>\n        <FromUserName><![CDATA[fromUser]]></FromUserName>\n        <CreateTime>1357290913</CreateTime>\n        <MsgType><![CDATA[voice]]></MsgType>\n        <MediaId><![CDATA[media_id]]></MediaId>\n        <Format><![CDATA[Format]]></Format>\n        <MsgId>1234567890123456</MsgId>\n        <AgentID>1</AgentID>\n        </xml>'
        msg = parse_message(xml)
        self.assertEqual('voice', msg.type)

    def test_parse_video_message(self):
        if False:
            i = 10
            return i + 15
        xml = '<xml>\n        <ToUserName><![CDATA[toUser]]></ToUserName>\n        <FromUserName><![CDATA[fromUser]]></FromUserName>\n        <CreateTime>1357290913</CreateTime>\n        <MsgType><![CDATA[video]]></MsgType>\n        <MediaId><![CDATA[media_id]]></MediaId>\n        <ThumbMediaId><![CDATA[thumb_media_id]]></ThumbMediaId>\n        <MsgId>1234567890123456</MsgId>\n        <AgentID>1</AgentID>\n        </xml>'
        msg = parse_message(xml)
        self.assertEqual('video', msg.type)

    def test_parse_location_message(self):
        if False:
            while True:
                i = 10
        xml = '<xml>\n        <ToUserName><![CDATA[toUser]]></ToUserName>\n        <FromUserName><![CDATA[fromUser]]></FromUserName>\n        <CreateTime>1351776360</CreateTime>\n        <MsgType><![CDATA[location]]></MsgType>\n        <Location_X>23.134521</Location_X>\n        <Location_Y>113.358803</Location_Y>\n        <Scale>20</Scale>\n        <Label><![CDATA[位置信息]]></Label>\n        <MsgId>1234567890123456</MsgId>\n        <AgentID>1</AgentID>\n        </xml>'
        msg = parse_message(xml)
        self.assertEqual('location', msg.type)

    def test_parse_link_message(self):
        if False:
            print('Hello World!')
        xml = '<xml>\n        <ToUserName><![CDATA[toUser]]></ToUserName>\n        <FromUserName><![CDATA[fromUser]]></FromUserName>\n        <CreateTime>1351776360</CreateTime>\n        <MsgType><![CDATA[link]]></MsgType>\n        <Title><![CDATA[公众平台官网链接]]></Title>\n        <Description><![CDATA[公众平台官网链接]]></Description>\n        <Url><![CDATA[url]]></Url>\n        <PicUrl><![CDATA[picurl]]></PicUrl>\n        <MsgId>1234567890123456</MsgId>\n        <AgentID>1</AgentID>\n        </xml>'
        msg = parse_message(xml)
        self.assertEqual('link', msg.type)

    def test_parse_subscribe_event(self):
        if False:
            return 10
        xml = '<xml>\n        <ToUserName><![CDATA[toUser]]></ToUserName>\n        <FromUserName><![CDATA[FromUser]]></FromUserName>\n        <CreateTime>123456789</CreateTime>\n        <MsgType><![CDATA[event]]></MsgType>\n        <Event><![CDATA[subscribe]]></Event>\n        <AgentID>1</AgentID>\n        </xml>'
        msg = parse_message(xml)
        self.assertEqual('event', msg.type)
        self.assertEqual('subscribe', msg.event)

    def test_parse_location_event(self):
        if False:
            for i in range(10):
                print('nop')
        xml = '<xml>\n        <ToUserName><![CDATA[toUser]]></ToUserName>\n        <FromUserName><![CDATA[fromUser]]></FromUserName>\n        <CreateTime>123456789</CreateTime>\n        <MsgType><![CDATA[event]]></MsgType>\n        <Event><![CDATA[LOCATION]]></Event>\n        <Latitude>23.137466</Latitude>\n        <Longitude>113.352425</Longitude>\n        <Precision>119.385040</Precision>\n        <AgentID>1</AgentID>\n        </xml>'
        msg = parse_message(xml)
        self.assertEqual('event', msg.type)
        self.assertEqual('location', msg.event)
        self.assertEqual(23.137466, msg.latitude)
        self.assertEqual(113.352425, msg.longitude)
        self.assertEqual(119.38504, msg.precision)

    def test_parse_click_event(self):
        if False:
            print('Hello World!')
        xml = '<xml>\n        <ToUserName><![CDATA[toUser]]></ToUserName>\n        <FromUserName><![CDATA[FromUser]]></FromUserName>\n        <CreateTime>123456789</CreateTime>\n        <MsgType><![CDATA[event]]></MsgType>\n        <Event><![CDATA[CLICK]]></Event>\n        <EventKey><![CDATA[EVENTKEY]]></EventKey>\n        <AgentID>1</AgentID>\n        </xml>'
        msg = parse_message(xml)
        self.assertEqual('event', msg.type)
        self.assertEqual('click', msg.event)
        self.assertEqual('EVENTKEY', msg.key)

    def test_parse_view_event(self):
        if False:
            for i in range(10):
                print('nop')
        xml = '<xml>\n        <ToUserName><![CDATA[toUser]]></ToUserName>\n        <FromUserName><![CDATA[FromUser]]></FromUserName>\n        <CreateTime>123456789</CreateTime>\n        <MsgType><![CDATA[event]]></MsgType>\n        <Event><![CDATA[VIEW]]></Event>\n        <EventKey><![CDATA[www.qq.com]]></EventKey>\n        <AgentID>1</AgentID>\n        </xml>'
        msg = parse_message(xml)
        self.assertEqual('event', msg.type)
        self.assertEqual('view', msg.event)
        self.assertEqual('www.qq.com', msg.url)

    def test_parse_unknown_message(self):
        if False:
            print('Hello World!')
        from wechatpy.messages import UnknownMessage
        xml = '<xml>\n        <ToUserName><![CDATA[toUser]]></ToUserName>\n        <FromUserName><![CDATA[fromUser]]></FromUserName>\n        <CreateTime>1348831860</CreateTime>\n        <MsgType><![CDATA[notsure]]></MsgType>\n        <MsgId>1234567890123456</MsgId>\n        <AgentID>1</AgentID>\n        </xml>'
        msg = parse_message(xml)
        self.assertTrue(isinstance(msg, UnknownMessage))

    def test_parse_modify_calendar(self):
        if False:
            for i in range(10):
                print('nop')
        xml = '\n        <xml>\n           <ToUserName><![CDATA[toUser]]></ToUserName>\n           <FromUserName><![CDATA[fromUser]]></FromUserName>\n           <CreateTime>1348831860</CreateTime>\n           <MsgType><![CDATA[event]]></MsgType>\n           <Event><![CDATA[modify_calendar]]></Event>\n           <CalId><![CDATA[wcjgewCwAAqeJcPI1d8Pwbjt7nttzAAA]]></CalId>\n        </xml>\n        '
        msg = parse_message(xml)
        self.assertIsInstance(msg, events.ModifyCalendarEvent)
        self.assertEqual('wcjgewCwAAqeJcPI1d8Pwbjt7nttzAAA', msg.calendar_id)

    def test_parse_delete_calendar(self):
        if False:
            for i in range(10):
                print('nop')
        xml = '\n        <xml>\n           <ToUserName><![CDATA[toUser]]></ToUserName>\n           <FromUserName><![CDATA[fromUser]]></FromUserName>\n           <CreateTime>1348831860</CreateTime>\n           <MsgType><![CDATA[event]]></MsgType>\n           <Event><![CDATA[delete_calendar]]></Event>\n           <CalId><![CDATA[wcjgewCwAAqeJcPI1d8Pwbjt7nttzAAA]]></CalId>\n        </xml>\n        '
        msg = parse_message(xml)
        self.assertIsInstance(msg, events.DeleteCalendarEvent)
        self.assertEqual('wcjgewCwAAqeJcPI1d8Pwbjt7nttzAAA', msg.calendar_id)

    def test_parse_add_schedule(self):
        if False:
            i = 10
            return i + 15
        xml = '\n        <xml>\n           <ToUserName><![CDATA[toUser]]></ToUserName>\n           <FromUserName><![CDATA[fromUser]]></FromUserName>\n           <CreateTime>1348831860</CreateTime>\n           <MsgType><![CDATA[event]]></MsgType>\n           <Event><![CDATA[add_schedule]]></Event>\n           <CalId><![CDATA[wcjgewCwAAqeJcPI1d8Pwbjt7nttzAAA]]></CalId>\n           <ScheduleId><![CDATA[17c7d2bd9f20d652840f72f59e796AAA]]></ScheduleId>\n        </xml>\n        '
        msg = parse_message(xml)
        self.assertIsInstance(msg, events.AddScheduleEvent)
        self.assertEqual('wcjgewCwAAqeJcPI1d8Pwbjt7nttzAAA', msg.calendar_id)
        self.assertEqual('17c7d2bd9f20d652840f72f59e796AAA', msg.schedule_id)

    def test_parse_modify_schedule(self):
        if False:
            while True:
                i = 10
        xml = '\n        <xml>\n           <ToUserName><![CDATA[toUser]]></ToUserName>\n           <FromUserName><![CDATA[fromUser]]></FromUserName>\n           <CreateTime>1348831860</CreateTime>\n           <MsgType><![CDATA[event]]></MsgType>\n           <Event><![CDATA[modify_schedule]]></Event>\n           <CalId><![CDATA[wcjgewCwAAqeJcPI1d8Pwbjt7nttzAAA]]></CalId>\n           <ScheduleId><![CDATA[17c7d2bd9f20d652840f72f59e796AAA]]></ScheduleId>\n        </xml>\n        '
        msg = parse_message(xml)
        self.assertIsInstance(msg, events.ModifyScheduleEvent)
        self.assertEqual('wcjgewCwAAqeJcPI1d8Pwbjt7nttzAAA', msg.calendar_id)
        self.assertEqual('17c7d2bd9f20d652840f72f59e796AAA', msg.schedule_id)

    def test_parse_delete_schedule(self):
        if False:
            while True:
                i = 10
        xml = '\n        <xml>\n           <ToUserName><![CDATA[toUser]]></ToUserName>\n           <FromUserName><![CDATA[fromUser]]></FromUserName>\n           <CreateTime>1348831860</CreateTime>\n           <MsgType><![CDATA[event]]></MsgType>\n           <Event><![CDATA[delete_schedule]]></Event>\n           <CalId><![CDATA[wcjgewCwAAqeJcPI1d8Pwbjt7nttzAAA]]></CalId>\n           <ScheduleId><![CDATA[17c7d2bd9f20d652840f72f59e796AAA]]></ScheduleId>\n        </xml>\n        '
        msg = parse_message(xml)
        self.assertIsInstance(msg, events.DeleteScheduleEvent)
        self.assertEqual('wcjgewCwAAqeJcPI1d8Pwbjt7nttzAAA', msg.calendar_id)
        self.assertEqual('17c7d2bd9f20d652840f72f59e796AAA', msg.schedule_id)

    def test_export(self):
        if False:
            for i in range(10):
                print('nop')
        xml = '\n        <xml>\n            <ToUserName><![CDATA[wx28dbb14e3720FAKE]]></ToUserName>\n            <FromUserName><![CDATA[FromUser]]></FromUserName>\n            <CreateTime>1425284517</CreateTime>\n            <MsgType><![CDATA[event]]></MsgType>\n            <Event><![CDATA[batch_job_result]]></Event>\n            <BatchJob>\n                <JobId><![CDATA[jobid_S0MrnndvRG5fadSlLwiBqiDDbM143UqTmKP3152FZk4]]></JobId>\n                <JobType><![CDATA[export_user]]></JobType>\n                <ErrCode>0</ErrCode>\n                <ErrMsg><![CDATA[ok]]></ErrMsg>\n            </BatchJob>\n        </xml>\n        '
        msg = parse_message(xml)
        self.assertIsInstance(msg, events.ExportEvent)
        self.assertEqual('jobid_S0MrnndvRG5fadSlLwiBqiDDbM143UqTmKP3152FZk4', msg.job_id)
        self.assertEqual('export_user', msg.job_type)
        xml = '\n        <xml>\n            <ToUserName><![CDATA[wx28dbb14e3720FAKE]]></ToUserName>\n            <FromUserName><![CDATA[FromUser]]></FromUserName>\n            <CreateTime>1425284517</CreateTime>\n            <MsgType><![CDATA[event]]></MsgType>\n            <Event><![CDATA[batch_job_result]]></Event>\n            <BatchJob>\n                <JobId><![CDATA[jobid_S0MrnndvRG5fadSlLwiBqiDDbM143UqTmKP3152FZk4]]></JobId>\n                <JobType><![CDATA[export_simple_user]]></JobType>\n                <ErrCode>0</ErrCode>\n                <ErrMsg><![CDATA[ok]]></ErrMsg>\n            </BatchJob>\n        </xml>\n        '
        msg = parse_message(xml)
        self.assertIsInstance(msg, events.ExportEvent)
        self.assertEqual('jobid_S0MrnndvRG5fadSlLwiBqiDDbM143UqTmKP3152FZk4', msg.job_id)
        self.assertEqual('export_simple_user', msg.job_type)
        xml = '\n        <xml>\n            <ToUserName><![CDATA[wx28dbb14e3720FAKE]]></ToUserName>\n            <FromUserName><![CDATA[FromUser]]></FromUserName>\n            <CreateTime>1425284517</CreateTime>\n            <MsgType><![CDATA[event]]></MsgType>\n            <Event><![CDATA[batch_job_result]]></Event>\n            <BatchJob>\n                <JobId><![CDATA[jobid_S0MrnndvRG5fadSlLwiBqiDDbM143UqTmKP3152FZk4]]></JobId>\n                <JobType><![CDATA[export_department]]></JobType>\n                <ErrCode>0</ErrCode>\n                <ErrMsg><![CDATA[ok]]></ErrMsg>\n            </BatchJob>\n        </xml>\n        '
        msg = parse_message(xml)
        self.assertIsInstance(msg, events.ExportEvent)
        self.assertEqual('jobid_S0MrnndvRG5fadSlLwiBqiDDbM143UqTmKP3152FZk4', msg.job_id)
        self.assertEqual('export_department', msg.job_type)
        xml = '\n        <xml>\n            <ToUserName><![CDATA[wx28dbb14e3720FAKE]]></ToUserName>\n            <FromUserName><![CDATA[FromUser]]></FromUserName>\n            <CreateTime>1425284517</CreateTime>\n            <MsgType><![CDATA[event]]></MsgType>\n            <Event><![CDATA[batch_job_result]]></Event>\n            <BatchJob>\n                <JobId><![CDATA[jobid_S0MrnndvRG5fadSlLwiBqiDDbM143UqTmKP3152FZk4]]></JobId>\n                <JobType><![CDATA[export_taguser]]></JobType>\n                <ErrCode>0</ErrCode>\n                <ErrMsg><![CDATA[ok]]></ErrMsg>\n            </BatchJob>\n        </xml>\n        '
        msg = parse_message(xml)
        self.assertIsInstance(msg, events.ExportEvent)
        self.assertEqual('jobid_S0MrnndvRG5fadSlLwiBqiDDbM143UqTmKP3152FZk4', msg.job_id)
        self.assertEqual('export_taguser', msg.job_type)

    def test_meeting(self):
        if False:
            while True:
                i = 10
        xml = '\n        <xml>\n            <ToUserName><![CDATA[toUser]]></ToUserName>\n            <FromUserName><![CDATA[fromUser]]></FromUserName>\n            <CreateTime>1348831860</CreateTime>\n            <MsgType><![CDATA[event]]></MsgType>\n            <Event><![CDATA[book_meeting_room]]></Event>\n            <MeetingRoomId>1</MeetingRoomId>\n            <MeetingId><![CDATA[mtebsada6e027c123cbafAAA]]></MeetingId>\n        </xml>\n        '
        msg = parse_message(xml)
        self.assertIsInstance(msg, events.BookMeetingRoom)
        self.assertEqual(1, msg.meeting_room_id)
        self.assertEqual('mtebsada6e027c123cbafAAA', msg.meeting_id)
        xml = '\n        <xml>\n            <ToUserName><![CDATA[toUser]]></ToUserName>\n            <FromUserName><![CDATA[fromUser]]></FromUserName>\n            <CreateTime>1348831860</CreateTime>\n            <MsgType><![CDATA[event]]></MsgType>\n            <Event><![CDATA[cancel_meeting_room]]></Event>\n            <MeetingId><![CDATA[mtebsada6e027c123cbafAAA]]></MeetingId>\n            <MeetingRoomId>1</MeetingRoomId>\n        </xml>\n        '
        msg = parse_message(xml)
        self.assertIsInstance(msg, events.CancelMeetingRoom)
        self.assertEqual(1, msg.meeting_room_id)
        self.assertEqual('mtebsada6e027c123cbafAAA', msg.meeting_id)