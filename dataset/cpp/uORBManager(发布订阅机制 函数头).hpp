/****************************************************************************

 uORB(Micro Object Request Broker,΢�������������)  ������Ϣ���� ���� ����
  ?uORB(Micro Object Request Broker,΢�������������)��PX4/Pixhawkϵͳ�зǳ���Ҫ�ҹؼ���һ��ģ�飬
  ���縺������ϵͳ�����ݴ����������еĴ��������ݡ�GPS��PPM�źŵȶ�Ҫ��оƬ��ȡ��ͨ��uORB���д��䵽
  ����ģ����м��㴦��ʵ����uORB��һ�׿硸���̡� ��IPCͨѶģ�顣��Pixhawk�У� 
  ���еĹ��ܱ������Խ���ģ��Ϊ��λ����ʵ�ֲ������������̼�����ݽ�������Ϊ��Ҫ��
  ����Ҫ�ܹ�����ʵʱ��������ص㡣 
??Pixhawkʹ�õ���NuttXʵʱARMϵͳ��uORBʵ�����Ƕ�����̴�ͬһ���豸�ļ���
���̼�ͨ�����ļ��ڵ�������ݽ����͹�������ͨ�������ġ����ߡ���������Ϣ��֮Ϊ�����⡹(topic)��
��Pixhawk �У�һ�����������һ����Ϣ���ͣ�ͨ�׵�����������͡�ÿ�����̿��ԡ����ġ����ߡ����������⣬
���Դ��ڶ�������ߣ�����һ�����̿��Զ��Ķ�����⣬����һ��������ʼ��ֻ��һ����Ϣ��

 ****************************************************************************/

#ifndef _uORBManager_hpp_
#define _uORBManager_hpp_

#include "uORBCommon.hpp"
#include "uORBDevices.hpp"
#include <stdint.h>
#ifdef __PX4_NUTTX
#include "ORBSet.hpp"
#else
#include <string>
#include <set>
#define ORBSet std::set<std::string>
#endif

#include "uORBCommunicator.hpp"

namespace uORB
{
class Manager;//�����ռ��µĶ����� 
}

/**
 * This is implemented as a singleton.  This class manages creating the
 * uORB nodes for each uORB topics and also implements the behavor of the
 * uORB Api's.
  �����ڵ� 
 */
class uORB::Manager  : public uORBCommunicator::IChannelRxHandler
{
public:
	// public interfaces for this class.

	/**
	 * Initialize the singleton. Call this before everything else.
	 * @return true on success
	 */
	static bool initialize();

	/**
	 * Method to get the singleton instance for the uORB::Manager.
	 * Make sure initialize() is called first.
	 * @return uORB::Manager*
	 */
	static uORB::Manager *get_instance()
	{
		return _Instance;
	}

	/**
	 * Get the DeviceMaster for a given Flavor. If it does not exist,
	 * it will be created and initialized.
	 * Note: the first call to this is not thread-safe.
	 * @return nullptr if initialization failed (and errno will be set)
	 */
	uORB::DeviceMaster *get_device_master(Flavor flavor);

	// ==== uORB interface methods ====
	/**
	 * Advertise as the publisher of a topic.
	 
	 *
	 * This performs the initial advertisement of a topic; it creates the topic
	 * node in /obj if required and publishes the initial data.
	 *
	 * Any number of advertisers may publish to a topic; publications are atomic
	 * but co-ordination between publishers is not provided by the ORB.
	 *
	 * Internally this will call orb_advertise_multi with an instance of 0 and
	 * default priority.
	 *
	 * @param meta    The uORB metadata (usually from the ORB_ID() macro)
	 *      for the topic.
	 * @param data    A pointer to the initial data to be published.
	 *      For topics updated by interrupt handlers, the advertisement
	 *      must be performed from non-interrupt context.
	 * @param queue_size  Maximum number of buffered elements. If this is 1, no queuing is
	 *      used.
	 * @return    nullptr on error, otherwise returns an object pointer
	 *      that can be used to publish to the topic.
	 *      If the topic in question is not known (due to an
	 *      ORB_DEFINE with no corresponding ORB_DECLARE)
	 *      this function will return nullptr and set errno to ENOENT.
	 */
	/*
���ܣ����淢���ߵ����⣻
˵�����ڷ�������֮ǰ�Ǳ���ģ�����������Ȼ�ܶ��ģ����ǵò������ݣ�
������
    meta:uORBԪ���󣬿�����Ϊ������id��һ����ͨ��ORB_ID(������)����ֵ;
    data:ָ��һ���ѱ���ʼ����������Ҫ���������ݴ洢������ָ�룻
����ֵ�������򷵻�ERROR;
        �ɹ��򷵻�һ�����Է�������ľ����  
		         ����  orb_publish(const struct orb_metadata *meta, orb_advert_t handle, const void *data) ; ������Ϣ���� 
		���������������û�ж����������᷵��-1��
		Ȼ��Ὣerrno��ֵΪENOENT;
eg:
    struct vehicle_attitude_s att; //����һ�� �뻰�� ����ϵ� �������ͱ��� 
    memset(&att, 0, sizeof(att));  // ��ʼ��Ϊ 0  
    int att_pub_fd = orb_advertise(ORB_ID(vehicle_attitude), &att);
	
	*/ 
	 //	 �㲥���� ������Ϣ  ������Ϣ  
	orb_advert_t orb_advertise(const struct orb_metadata *meta, const void *data, unsigned int queue_size = 1)
	{
		return orb_advertise_multi(meta, data, nullptr, ORB_PRIO_DEFAULT, queue_size);
	}

	/**
	 * Advertise as the publisher of a topic.
	 *
	 * This performs the initial advertisement of a topic; it creates the topic
	 * node in /obj if required and publishes the initial data.
	 *
	 * Any number of advertisers may publish to a topic; publications are atomic
	 * but co-ordination between publishers is not provided by the ORB.
	 *
	 * The multi can be used to create multiple independent instances of the same topic
	 * (each instance has its own buffer).
	 * This is useful for multiple publishers who publish the same topic. The subscriber
	 * then subscribes to all instances and chooses which source he wants to use.
	 *
	 * @param meta    The uORB metadata (usually from the ORB_ID() macro)
	 *      for the topic.
	 * @param data    A pointer to the initial data to be published.
	 *      For topics updated by interrupt handlers, the advertisement
	 *      must be performed from non-interrupt context.
	 * @param instance  Pointer to an integer which will yield the instance ID (0-based)
	 *      of the publication. This is an output parameter and will be set to the newly
	 *      created instance, ie. 0 for the first advertiser, 1 for the next and so on.
	 * @param priority  The priority of the instance. If a subscriber subscribes multiple
	 *      instances, the priority allows the subscriber to prioritize the best
	 *      data source as long as its available. The subscriber is responsible to check
	 *      and handle different priorities (@see orb_priority()).
	 * @param queue_size  Maximum number of buffered elements. If this is 1, no queuing is
	 *      used.
	 * @return    ERROR on error, otherwise returns a handle
	 *      that can be used to publish to the topic.
	 *      If the topic in question is not known (due to an
	 *      ORB_DEFINE with no corresponding ORB_DECLARE)
	 *      this function will return -1 and set errno to ENOENT.
	 */
	 
	/*
���ܣ��豸/�������Ķ��ʵ��ʵ�ֹ��棬���ô˺�������ע�������Ƶ���������
˵���������ڷ��������ж����ͬ�Ĵ������������ǵ��������������ƣ�����Ҫע�Ἰ����ͬ�Ļ��⣻
������
    meta:uORBԪ���󣬿�����Ϊ������id��һ����ͨ��ORB_ID(������)����ֵ;
    data:ָ��һ���ѱ���ʼ����������Ҫ���������ݴ洢������ָ�룻
    instance:����ָ�룬ָ��ʵ����ID����0��ʼ����
    priority:ʵ�������ȼ�������û����Ķ��ʵ�������ȼ����趨����ʹ�û�ʹ�����ȼ��ߵ���������Դ��
����ֵ��
    �����򷵻�ERROR;�ɹ��򷵻�һ�����Է�������ľ����
	���������������û�ж����������᷵��-1��Ȼ��Ὣerrno��ֵΪENOENT;
eg:
    struct orb_test t;
    t.val = 0;
    int instance0;
    orb_advert_t pfd0 = orb_advertise_multi(ORB_ID(orb_multitest), &t, &instance0, ORB_PRIO_MAX);
    
	*/ 
    //�㲥���� ������Ϣ  �����Ϣ ������ ��ͬ�ķ����ڵ� ������ͬ����ʽ  ���ڲ�ͬ�� ������������������ ��ѡ���ԵĶ��ģ� 
	orb_advert_t orb_advertise_multi(const struct orb_metadata *meta, const void *data, int *instance,
					 int priority, unsigned int queue_size = 1) ;


	/**
	 * Unadvertise a topic.
	 
	 * 
	 
	 * @param handle  handle returned by orb_advertise or orb_advertise_multi.
	 * @return 0 on success
	 */

    
	 // ȡ����������  
	int orb_unadvertise(orb_advert_t handle);

	/**
	 * Publish new data to a topic.
	 
 
	 
	 *
	 * The data is atomically published to the topic and any waiting subscribers
	 * will be notified.  Subscribers that are not waiting can check the topic
	 * for updates using orb_check and/or orb_stat.
	 *
	 * @param meta    The uORB metadata (usually from the ORB_ID() macro)
	 *      for the topic.
	 * @handle    The handle returned from orb_advertise.
	 * @param data    A pointer to the data to be published.
	 * @return    OK on success, ERROR otherwise with errno set accordingly.
	 */
	 
	 
	/*
���ܣ����������ݵ����⣻
������
    meta:uORBԪ���󣬿�����Ϊ������id��һ����ͨ��ORB_ID(������)����ֵ;
    handle:orb_advertise�������صľ������   int att_pub_fd = orb_advertise(ORB_ID(vehicle_attitude), &att);
    data:ָ����������ݵ�ָ�룻
����ֵ��OK��ʾ�ɹ������󷵻�ERROR���������и��ݵ�ȥ����errno;
eg: 
              ����ǰ��Ҫ ������Ϣ���� 
				att.q[0] = raw.accelerometer_m_s2[0];
				att.q[1] = raw.accelerometer_m_s2[1];
				att.q[2] = raw.accelerometer_m_s2[2];
    orb_publish(ORB_ID(vehicle_attitude), att_pub_fd, &att);
	*/ 
   // �����Ϸ����µ���Ϣ 
	int  orb_publish(const struct orb_metadata *meta, orb_advert_t handle, const void *data) ;

	/**
	 * Subscribe to a topic.
	 *
	 * The returned value is a file descriptor that can be passed to poll()
	 * in order to wait for updates to a topic, as well as topic_read,
	 * orb_check and orb_stat.
	 *
	 * Subscription will succeed even if the topic has not been advertised;
	 * in this case the topic will have a timestamp of zero, it will never
	 * signal a poll() event, checking will always return false and it cannot
	 * be copied. When the topic is subsequently advertised, poll, check,
	 * stat and copy calls will react to the initial publication that is
	 * performed as part of the advertisement.
	 *
	 * Subscription will fail if the topic is not known to the system, i.e.
	 * there is nothing in the system that has declared the topic and thus it
	 * can never be published.
	 *
	 * Internally this will call orb_subscribe_multi with instance 0.
	 *
	 * @param meta    The uORB metadata (usually from the ORB_ID() macro)
	 *      for the topic.
	 * @return    ERROR on error, otherwise returns a handle
	 *      that can be used to read and update the topic.
	 */
	 /*
���ܣ��������⣨topic��;
˵������ʹ���ĵ�����û�б����棬����Ҳ�ܶ��ĳɹ�����������������£�ȴ�ò������ݣ�ֱ�����ⱻ���棻
������
        meta:uORBԪ���󣬿�����Ϊ������id��һ����ͨ��ORB_ID(������)����ֵ��
����ֵ��
      �����򷵻�ERROR;�ɹ��򷵻�һ�����Զ�ȡ���ݡ����»���ľ����
	  ��������ĵ�����û�ж����������᷵��-1��Ȼ��Ὣerrno��ֵΪENOENT;
eg:
    int fd = orb_subscribe(ORB_ID(topicName));
	 */
	 // ���Ļ���  �ϵĵ�����Ϣ���� 
	int  orb_subscribe(const struct orb_metadata *meta) ;
     

	/**
	 * Subscribe to a multi-instance of a topic.
	 *
	 * The returned value is a file descriptor that can be passed to poll()
	 * in order to wait for updates to a topic, as well as topic_read,
	 * orb_check and orb_stat.
	 *
	 * Subscription will succeed even if the topic has not been advertised;
	 * in this case the topic will have a timestamp of zero, it will never
	 * signal a poll() event, checking will always return false and it cannot
	 * be copied. When the topic is subsequently advertised, poll, check,
	 * stat and copy calls will react to the initial publication that is
	 * performed as part of the advertisement.
	 *
	 * Subscription will fail if the topic is not known to the system, i.e.
	 * there is nothing in the system that has declared the topic and thus it
	 * can never be published.
	 *
	 * If a publisher publishes multiple instances the subscriber should
	 * subscribe to each instance with orb_subscribe_multi
	 * (@see orb_advertise_multi()).
	 *
	 * @param meta    The uORB metadata (usually from the ORB_ID() macro)
	 *      for the topic.
	 * @param instance  The instance of the topic. Instance 0 matches the
	 *      topic of the orb_subscribe() call, higher indices
	 *      are for topics created with orb_advertise_multi().
	 * @return    ERROR on error, otherwise returns a handle
	 *      that can be used to read and update the topic.
	 *      If the topic in question is not known (due to an
	 *      ORB_DEFINE_OPTIONAL with no corresponding ORB_DECLARE)
	 *      this function will return -1 and set errno to ENOENT.
	 */
	
 /*
 ���ܣ��������⣨topic��;
˵����ͨ��ʵ����ID������ȷ����������ĸ�ʵ����
������
    meta:uORBԪ���󣬿�����Ϊ������id��һ����ͨ��ORB_ID(������)����ֵ;
    instance:����ʵ��ID;ʵ��ID=0��orb_subscribe()ʵ����ͬ��
����ֵ��
    �����򷵻�ERROR;�ɹ��򷵻�һ�����Զ�ȡ���ݡ����»���ľ������������ĵ�����û�ж����������᷵��-1��Ȼ��Ὣerrno��ֵΪENOENT;
eg:
    int sfd1 = orb_subscribe_multi(ORB_ID(orb_multitest), 1);
 */ 
 // ���Ļ���  �ϵĶ����Ϣ���� 
	int  orb_subscribe_multi(const struct orb_metadata *meta, unsigned instance) ;

	/**
	 * Unsubscribe from a topic.
	 *
	 * @param handle  A handle returned from orb_subscribe.
	 * @return    OK on success, ERROR otherwise with errno set accordingly.
	 */
	 	 /*
���ܣ�ȡ���������⣻
������
    handle:��������
����ֵ��
    OK��ʾ�ɹ������󷵻�ERROR;�������и��ݵ�ȥ����errno;
eg:
	 int fd = orb_subscribe(ORB_ID(topicName));
	 
    ret = orb_unsubscribe(fd);
	*/
 // ȡ�����Ļ��� 
	int  orb_unsubscribe(int handle) ;

	/**
	 * Fetch data from a topic.
	 *
	 * This is the only operation that will reset the internal marker that
	 * indicates that a topic has been updated for a subscriber. Once poll
	 * or check return indicating that an updaet is available, this call
	 * must be used to update the subscription.
	 *
	 * @param meta    The uORB metadata (usually from the ORB_ID() macro)
	 *      for the topic.
	 * @param handle  A handle returned from orb_subscribe.
	 * @param buffer  Pointer to the buffer receiving the data, or NULL
	 *      if the caller wants to clear the updated flag without
	 *      using the data.
	 * @return    OK on success, ERROR otherwise with errno set accordingly.
	 */
	 /*
	 ���ܣ��Ӷ��ĵ������л�ȡ���ݲ������ݱ��浽buffer�У�
������
    meta:uORBԪ���󣬿�����Ϊ������id��һ����ͨ��ORB_ID(������)����ֵ;
    handle:�������ⷵ�صľ����  ��   int fd = orb_subscribe(ORB_ID(topicName));  topicName����Ϊ sensor_combined
    buffer:�������л�ȡ�����ݣ�
����ֵ��
    ����OK��ʾ��ȡ���ݳɹ������󷵻�ERROR;�������и��ݵ�ȥ����errno;
eg:
    struct sensor_combined_s raw;
    orb_copy(ORB_ID(sensor_combined), sensor_sub_fd, &raw);
	 */
// �ӻ����ϵõ���Ϣ  ���Ƶ�����  �� �ڴ������ 
	int  orb_copy(const struct orb_metadata *meta, int handle, void *buffer) ;

	/**
	 * Check whether a topic has been published to since the last orb_copy.
	 *
	 * This check can be used to determine whether to copy the topic when
	 * not using poll(), or to avoid the overhead of calling poll() when the
	 * topic is likely to have updated.
	 *
	 * Updates are tracked on a per-handle basis; this call will continue to
	 * return true until orb_copy is called using the same handle. This interface
	 * should be preferred over calling orb_stat due to the race window between
	 * stat and copy that can lead to missed updates.
	 *
	 * @param handle  A handle returned from orb_subscribe.
	 * @param updated Set to true if the topic has been updated since the
	 *      last time it was copied using this handle.
	 * @return    OK if the check was successful, ERROR otherwise with
	 *      errno set accordingly.
	 */
/*
	���ܣ������߿����������һ�������ڷ�������һ�θ������ݺ���û�ж����ߵ��ù�ob_copy�����ա��������
˵��������������ڱ�����ǰ�����˶��ģ���ô���API�����ء�not-updated��ֱ�����ⱻ���档���Բ���poll��ֻ���������ʵ�����ݵĻ�ȡ��
������
    handle:��������
    updated:��������һ�θ��µ����ݱ���ȡ�ˣ���⵽������updatedΪture;
����ֵ��
    OK��ʾ���ɹ������󷵻�ERROR;�������и��ݵ�ȥ����errno;
eg:
    if (PX4_OK != orb_check(sfd, &updated)) {
        return printf("check(1) failed");
    }
    if (updated) {
        return printf("spurious updated flag");
    }
 //or

    bool updated;
    struct random_integer_data rd;

    //check to see whether the topic has updated since the last time we read it /
    orb_check(topic_handle, &updated);

    if (updated) {
       //make a local copy of the updated data structure/
        orb_copy(ORB_ID(random_integer), topic_handle, &rd);
        printf("Random integer is now %d\n", rd.r);
    } 
*/

// �����ϴ�ȡ��Ϣ��  
	int  orb_check(int handle, bool *updated) ;

	/**
	 * Return the last time that the topic was updated. If a queue is used, it returns
	 * the timestamp of the latest element in the queue.
	 *
	 * @param handle  A handle returned from orb_subscribe.
	 * @param time    Returns the absolute time that the topic was updated, or zero if it has
	 *      never been updated. Time is measured in microseconds.
	 * @return    OK on success, ERROR otherwise with errno set accordingly.
	 */

/*
���ܣ������߿����������һ���������ķ���ʱ�䣻
������
    handle:��������
    time:���������󷢲���ʱ�䣻0��ʾ������û�з����򹫸棻
����ֵ��
    OK��ʾ���ɹ������󷵻�ERROR;�������и��ݵ�ȥ����errno;
eg:
    ret = orb_stat(handle,time);
*/
 // ���� ���Ļ��� �ϴθ��·�����Ϣ ��ʱ�� 
	int  orb_stat(int handle, uint64_t *time) ;

	/**
	 * Check if a topic has already been created (a publisher or a subscriber exists with
	 * the given instance).
	 *
	 * @param meta    ORB topic metadata.
	 * @param instance  ORB instance
	 * @return    OK if the topic exists, ERROR otherwise with errno set accordingly.
	 */
	 
	 
	 /*
	���ܣ����һ�������Ƿ���ڣ�
������
    meta:uORBԪ���󣬿�����Ϊ������id��һ����ͨ��ORB_ID(������)����ֵ;
    instance:ORB ʵ��ID;
����ֵ��
    OK��ʾ���ɹ������󷵻�ERROR;�������и��ݵ�ȥ����errno;
eg:
    ret = orb_exists(ORB_ID(vehicle_attitude),0); 
	 */
// ��� �����Ƿ���� �Ƿ��Ѿ������� 
	int  orb_exists(const struct orb_metadata *meta, int instance) ;

	/**
	 * Return the priority of the topic
	 *
	 * @param handle  A handle returned from orb_subscribe.
	 * @param priority  Returns the priority of this topic. This is only relevant for
	 *      topics which are published by multiple publishers (e.g. mag0, mag1, etc.)
	 *      and allows a subscriber to pick the topic with the highest priority,
	 *      independent of the startup order of the associated publishers.
	 * @return    OK on success, ERROR otherwise with errno set accordingly.
	 */
	
	/*
	���ܣ���ȡ�������ȼ���
������
    handle:��������
    priority:��Ż�ȡ�����ȼ���
����ֵ��
    OK��ʾ���ɹ������󷵻�ERROR;�������и��ݵ�ȥ����errno;
eg:
    ret = orb_priority(handle,&priority);
    
	*/ 
//  һ������  ���з����ڵ㣨�ߣ� �����ȴ��򣨶��ڻ��ⶩ���ߣ� 
	int  orb_priority(int handle, int32_t *priority) ;

	/**
	 * Set the minimum interval between which updates are seen for a subscription.
	 *
	 * If this interval is set, the subscriber will not see more than one update
	 * within the period.
	 *
	 * Specifically, the first time an update is reported to the subscriber a timer
	 * is started. The update will continue to be reported via poll and orb_check, but
	 * once fetched via orb_copy another update will not be reported until the timer
	 * expires.
	 *
	 * This feature can be used to pace a subscriber that is watching a topic that
	 * would otherwise update too quickly.
	 *
	 * @param handle  A handle returned from orb_subscribe.
	 * @param interval  An interval period in milliseconds.
	 * @return    OK on success, ERROR otherwise with ERRNO set accordingly.
	 */
	/*
	
���ܣ����ö��ĵ���Сʱ������
˵������������ˣ����������ڷ��������ݽ����Ĳ�������Ҫע����ǣ����ú󣬵�һ�ε����ݶ��Ļ�����������õ�Ƶ������ȡ��
������
    handle:orb_subscribe�������صľ����
    interval:���ʱ�䣬��λms;
����ֵ��OK��ʾ�ɹ������󷵻�ERROR���������и��ݵ�ȥ����errno;
eg:
	int sensor_sub_fd = orb_subscribe(ORB_ID(sensor_combined));
    orb_set_interval(sensor_sub_fd, 1000);
    
	*/ 
// ���� ������   orb_copy ȡ��������Ϣ��  ��Сʱ����  
	int  orb_set_interval(int handle, unsigned interval) ;


	/**
	 * Get the minimum interval between which updates are seen for a subscription.
	 *
	 * @see orb_set_interval()
	 *
	 * @param handle  A handle returned from orb_subscribe.
	 * @param interval  The returned interval period in milliseconds.
	 * @return    OK on success, ERROR otherwise with ERRNO set accordingly.
	 */
// �õ� ������   orb_copy ȡ��������Ϣ��  ʱ����  
	int	orb_get_interval(int handle, unsigned *interval);

	/**
	 * Method to set the uORBCommunicator::IChannel instance.
	 * @param comm_channel
	 *  The IChannel instance to talk to remote proxies.
	 * @note:
	 *  Currently this call only supports the use of one IChannel
	 *  Future extensions may include more than one IChannel's.
	 */
	void set_uorb_communicator(uORBCommunicator::IChannel *comm_channel);

	/**
	 * Gets the uORB Communicator instance.
	 */
	uORBCommunicator::IChannel *get_uorb_communicator(void);

	/**
	 * Utility method to check if there is a remote subscriber present
	 * for a given topic
	 */
	bool is_remote_subscriber_present(const char *messageName);

private: // class methods
	/**
	 * Advertise a node; don't consider it an error if the node has
	 * already been advertised.
	 *
	 * @todo verify that the existing node is the same as the one
	 *       we tried to advertise.
	 */
	int
	node_advertise
	(
		const struct orb_metadata *meta,
		int *instance = nullptr,
		int priority = ORB_PRIO_DEFAULT
	);

	/**
	 * Common implementation for orb_advertise and orb_subscribe.
	 *
	 * Handles creation of the object and the initial publication for
	 * advertisers.
	 */
	int
	node_open
	(
		Flavor f,
		const struct orb_metadata *meta,
		const void *data,
		bool advertiser,
		int *instance = nullptr,
		int priority = ORB_PRIO_DEFAULT
	);

private: // data members
	static Manager *_Instance;
	// the communicator channel instance.
	uORBCommunicator::IChannel *_comm_channel;
	ORBSet _remote_subscriber_topics;
	ORBSet _remote_topics;

	DeviceMaster *_device_masters[Flavor_count]; ///< Allow at most one DeviceMaster per Flavor

private: //class methods
	Manager();
	~Manager();

	/**
	 * Interface to process a received topic from remote.
	 * @param topic_name
	 * 	This represents the uORB message Name (topic); This message Name should be
	 * 	globally unique.
	 * @param isAdvertisement
	 * 	Represents if the topic has been advertised or is no longer avialable.
	 * @return
	 *  0 = success; This means the messages is successfully handled in the
	 *  	handler.
	 *  otherwise = failure.
	 */
	virtual int16_t process_remote_topic(const char *topic_name, bool isAdvertisement);

	/**
	   * Interface to process a received AddSubscription from remote.
	   * @param messageName
	   *  This represents the uORB message Name; This message Name should be
	   *  globally unique.
	   * @param msgRate
	   *  The max rate at which the subscriber can accept the messages.
	   * @return
	   *  0 = success; This means the messages is successfully handled in the
	   *    handler.
	   *  otherwise = failure.
	   */
	virtual int16_t process_add_subscription(const char *messageName,
			int32_t msgRateInHz);

	/**
	 * Interface to process a received control msg to remove subscription
	 * @param messageName
	 *  This represents the uORB message Name; This message Name should be
	 *  globally unique.
	 * @return
	 *  0 = success; This means the messages is successfully handled in the
	 *    handler.
	 *  otherwise = failure.
	 */
	virtual int16_t process_remove_subscription(const char *messageName);

	/**
	 * Interface to process the received data message.
	 * @param messageName
	 *  This represents the uORB message Name; This message Name should be
	 *  globally unique.
	 * @param length
	 *  The length of the data buffer to be sent.
	 * @param data
	 *  The actual data to be sent.
	 * @return
	 *  0 = success; This means the messages is successfully handled in the
	 *    handler.
	 *  otherwise = failure.
	 */
	virtual int16_t process_received_message(const char *messageName,
			int32_t length, uint8_t *data);


#ifdef ORB_USE_PUBLISHER_RULES

	struct PublisherRule {
		const char **topics; //null-terminated list of topic names
		const char *module_name; //only this module is allowed to publish one of the topics
		bool ignore_other_topics;
	};

	/**
	 * test if str starts with pre
	 */
	bool startsWith(const char *pre, const char *str);

	/**
	 * find a topic in a rule
	 */
	bool findTopic(const PublisherRule &rule, const char *topic_name);

	/**
	 * trim whitespace from the beginning of a string
	 */
	void strTrim(const char **str);

	/**
	 * Read publisher rules from a file. It has the format:
	 *
	 * restrict_topics: <topic1>, <topic2>, <topic3>
	 * module: <module_name>
	 * [ignore_others:true]
	 *
	 * @return 0 on success, <0 otherwise
	 */
	int readPublisherRulesFromFile(const char *file_name, PublisherRule &rule);

	PublisherRule _publisher_rule;
	bool _has_publisher_rules = false;

#endif /* ORB_USE_PUBLISHER_RULES */

};

#endif /* _uORBManager_hpp_ */
