//
// Copyright (C) 2007 Pingtel Corp., certain elements licensed under a Contributor Agreement.
// Contributors retain copyright to elements licensed under a Contributor Agreement.
// Licensed to the User under the LGPL license.
//
//
// $$
////////////////////////////////////////////////////////////////////////

#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/TestCase.h>

#include "utl/UtlString.h"
#include "os/OsConfigDb.h"
#include "utl/TestPlugin.h"
#include "utl/PluginHooks.h"
#include <sipxunit/TestUtilities.h>
#include "config.h"

#ifndef PLUGIN_EXT
#  ifdef LT_MODULE_EXT
#    define PLUGIN_EXT LT_MODULE_EXT
#  else
#    define PLUGIN_EXT LTDL_SHLIB_EXT
#  endif
#endif

using namespace std ;

#define PLUGIN_LIB_DIR TEST_DIR "/testplugin/.libs/"

class PluginHooksTest : public CppUnit::TestCase
{
   CPPUNIT_TEST_SUITE(PluginHooksTest);
   CPPUNIT_TEST(testNoHooks);
   CPPUNIT_TEST(testOneHook);
   CPPUNIT_TEST(testReadConfig);
   CPPUNIT_TEST(testTwoInstances);
   CPPUNIT_TEST(testTwoHookTypes);
   CPPUNIT_TEST(testReconfigure);
   CPPUNIT_TEST(testNoLibrary);
   CPPUNIT_TEST_SUITE_END();

public:

   void testNoHooks()
      {
         OsConfigDb configuration;

         configuration.set("NOHOOKS_NOTPASSED_HOOK_LIBRARY.Error", PLUGIN_LIB_DIR "libfoo" PLUGIN_EXT);
         configuration.set("NOHOOKS_OTHERPARAM", "DummyValue");

         // there are no hooks configured for this prefix
         PluginHooks testPlugins("getTestPlugin", "NOHOOKS_PASSED");
         testPlugins.readConfig(configuration);

         PluginIterator shouldBeEmpty(testPlugins);

         UtlString name;

         // confirm that there are no hooks configured.
         CPPUNIT_ASSERT(shouldBeEmpty.next(&name) == NULL);
         CPPUNIT_ASSERT(name.isNull());

         CPPUNIT_ASSERT(shouldBeEmpty.next() == NULL);
      }

   void testOneHook()
      {
         OsConfigDb configuration;

         configuration.set("ONEHOOK_NOTPASSED_HOOK_LIBRARY.Error", PLUGIN_LIB_DIR "libfoo" PLUGIN_EXT);
         configuration.set("ONEHOOK_OTHERPARAM", "DummyValue");

         // configure one hook, with no parameters
         configuration.set("ONEHOOK_TEST_HOOK_LIBRARY.Only",
                           PLUGIN_LIB_DIR "libtestpluginA" PLUGIN_EXT);

         PluginHooks testPlugins("getTestPlugin", "ONEHOOK_TEST");
         testPlugins.readConfig(configuration);

         PluginIterator plugin(testPlugins);

         UtlString name;
         TestPlugin* thePlugin;

         // get that one hook
         thePlugin = static_cast<TestPlugin*>(plugin.next(&name));

         // confirm that it is loaded and that it has the right instance name
         CPPUNIT_ASSERT(thePlugin != NULL);
         CPPUNIT_ASSERT(!name.isNull());
         ASSERT_STR_EQUAL("Only",name.data());

         // invoke the pluginName method from the hook to confirm that it's there
         UtlString fullName;
         thePlugin->pluginName(fullName);
         ASSERT_STR_EQUAL("TestPluginA::Only",fullName.data());

         // confirm that it is the only configured hook
         CPPUNIT_ASSERT(plugin.next(&name) == NULL);
         CPPUNIT_ASSERT(name.isNull());
      }

   void testReadConfig()
      {
         OsConfigDb configuration;

         configuration.set("READCONFIG_NOTPASSED_HOOK_LIBRARY.Error", PLUGIN_LIB_DIR "libfoo" PLUGIN_EXT);
         configuration.set("READCONFIG_OTHERPARAM", "DummyValue");

         // configure a hook
         configuration.set("READCONFIG_TEST_HOOK_LIBRARY.Only",
                           PLUGIN_LIB_DIR "libtestpluginA" PLUGIN_EXT);
         // with two configuration values
         configuration.set("READCONFIG_TEST.Only.VALUE1", "FirstValue");
         configuration.set("READCONFIG_TEST.Only.VALUE2", "SecondValue");

         PluginHooks testPlugins("getTestPlugin", "READCONFIG_TEST");
         testPlugins.readConfig(configuration);

         PluginIterator plugin(testPlugins);

         UtlString name;
         TestPlugin* thePlugin;

         // get the hook and confirm the name
         thePlugin = static_cast<TestPlugin*>(plugin.next(&name));
         CPPUNIT_ASSERT(thePlugin != NULL);
         CPPUNIT_ASSERT(!name.isNull());
         ASSERT_STR_EQUAL("Only",name.data());
         UtlString fullName;
         thePlugin->pluginName(fullName);
         ASSERT_STR_EQUAL("TestPluginA::Only",fullName.data());

         // read the two parameters and confirm that the hook got them
         UtlString pluginValue;
         CPPUNIT_ASSERT(thePlugin->getConfiguredValueFor("VALUE1", pluginValue));
         ASSERT_STR_EQUAL("FirstValue",pluginValue.data());

         CPPUNIT_ASSERT(thePlugin->getConfiguredValueFor("VALUE2", pluginValue));
         ASSERT_STR_EQUAL("SecondValue",pluginValue.data());

         // try some bogus parameter
         CPPUNIT_ASSERT(!thePlugin->getConfiguredValueFor("UNCONFIGURED", pluginValue));

         // confirm that this is the only hook
         CPPUNIT_ASSERT(plugin.next(&name) == NULL);
         CPPUNIT_ASSERT(name.isNull());
      }

   void testTwoInstances()
      {
         OsConfigDb configuration;

         configuration.set("TWO_INST_NOTPASSED_HOOK_LIBRARY.Error", PLUGIN_LIB_DIR "libfoo" PLUGIN_EXT);
         configuration.set("TWO_INST_OTHERPARAM", "DummyValue");

         // configure two instances of the same hook library, with different parameters
         configuration.set("TWO_INST_TEST_HOOK_LIBRARY.First",
                           PLUGIN_LIB_DIR "libtestpluginA" PLUGIN_EXT);
         configuration.set("TWO_INST_TEST.First.VALUE", "FirstValue");

         configuration.set("TWO_INST_TEST_HOOK_LIBRARY.Second",
                           PLUGIN_LIB_DIR "libtestpluginA" PLUGIN_EXT);
         configuration.set("TWO_INST_TEST.Second.VALUE", "SecondValue");

         // load up the hooks
         PluginHooks testPlugins("getTestPlugin", "TWO_INST_TEST");
         testPlugins.readConfig(configuration);
         PluginIterator plugin(testPlugins);

         UtlString name;
         TestPlugin* thePlugin;

         // get the first instance and check its name
         thePlugin = static_cast<TestPlugin*>(plugin.next(&name));
         CPPUNIT_ASSERT(thePlugin != NULL);
         CPPUNIT_ASSERT(!name.isNull());
         ASSERT_STR_EQUAL("First",name.data());
         UtlString fullName;
         thePlugin->pluginName(fullName);
         ASSERT_STR_EQUAL("TestPluginA::First",fullName.data());

         const UtlString ValueKey("VALUE");

         // check that the first instance is using the correct configuration value
         UtlString pluginValue;
         CPPUNIT_ASSERT(thePlugin->getConfiguredValueFor(ValueKey, pluginValue));
         ASSERT_STR_EQUAL("FirstValue",pluginValue.data());

         // get the second instance and confirm its name
         thePlugin = static_cast<TestPlugin*>(plugin.next(&name));
         CPPUNIT_ASSERT(thePlugin != NULL);
         CPPUNIT_ASSERT(!name.isNull());
         ASSERT_STR_EQUAL("Second",name.data());
         thePlugin->pluginName(fullName);
         ASSERT_STR_EQUAL("TestPluginA::Second",fullName.data());

         // check that the second instance is using the correct configuration value
         CPPUNIT_ASSERT(thePlugin->getConfiguredValueFor(ValueKey, pluginValue));
         ASSERT_STR_EQUAL("SecondValue",pluginValue.data());

         // and make sure that is the end of the iteration
         CPPUNIT_ASSERT(plugin.next(&name) == NULL);
         CPPUNIT_ASSERT(name.isNull());
      }

   void testTwoHookTypes()
      {
         OsConfigDb configuration;

         configuration.set("TWO_TYPE_NOTPASSED_HOOK_LIBRARY.Error", PLUGIN_LIB_DIR "libfoo" PLUGIN_EXT);
         configuration.set("TWO_TYPE_OTHERPARAM", "DummyValue");

         // Configure two different hook types - each in its own library
         //   (we cheat - it's the same source, modified by configuration
         //    switches; see comments in ../testplugin/TestPlugin.cpp)
         configuration.set("TWO_TYPE1_HOOK_LIBRARY.First",
                           PLUGIN_LIB_DIR "libtestpluginA" PLUGIN_EXT);
         configuration.set("TWO_TYPE1.First.VALUE", "FirstValue");

         configuration.set("TWO_TYPE2_HOOK_LIBRARY.Second",
                           PLUGIN_LIB_DIR "libtestpluginB" PLUGIN_EXT);
         configuration.set("TWO_TYPE2.Second.VALUE", "SecondValue");

         // load the first hook type
         PluginHooks test1Plugins("getTestPlugin", "TWO_TYPE1");
         test1Plugins.readConfig(configuration);

         // load the second hook type
         PluginHooks test2Plugins("getTestPlugin", "TWO_TYPE2");
         test2Plugins.readConfig(configuration);

         // create iterators for both hook types
         PluginIterator plugin1(test1Plugins);
         PluginIterator plugin2(test2Plugins);

         UtlString name;
         TestPlugin* thePlugin;
         UtlString fullName;

         // get the first instance of the first hook type and confirm the names
         thePlugin = static_cast<TestPlugin*>(plugin1.next(&name));
         CPPUNIT_ASSERT(thePlugin != NULL);
         thePlugin->pluginName(fullName);
         ASSERT_STR_EQUAL("TestPluginA::First",fullName.data());

         // confirm the first hook type is using the correct configuration value
         const UtlString ValueKey("VALUE");
         UtlString pluginValue;
         CPPUNIT_ASSERT(thePlugin->getConfiguredValueFor(ValueKey, pluginValue));
         ASSERT_STR_EQUAL("FirstValue",pluginValue.data());

         // get the first instance of the second hook type and confim its names
         thePlugin = static_cast<TestPlugin*>(plugin2.next(&name));
         CPPUNIT_ASSERT(thePlugin != NULL);
         thePlugin->pluginName(fullName);
         ASSERT_STR_EQUAL("TestPluginB::Second",fullName.data());

         // confirm the configuration of the second hook type
         CPPUNIT_ASSERT(thePlugin->getConfiguredValueFor(ValueKey, pluginValue));
         ASSERT_STR_EQUAL("SecondValue",pluginValue.data());

         // make sure each type only has the one instance
         CPPUNIT_ASSERT(plugin1.next(&name) == NULL);
         CPPUNIT_ASSERT(name.isNull());
         CPPUNIT_ASSERT(plugin2.next(&name) == NULL);
         CPPUNIT_ASSERT(name.isNull());
      }

   void testReconfigure()
      {
         PluginHooks testPlugins("getTestPlugin", "RECONFIG_TEST");

         const UtlString ValueKey("VALUE");

         UtlString name;
         TestPlugin* thePlugin;
         UtlString pluginValue;
         UtlString fullName;


         {
            // Do the initial configuration of two instances of a hook
            OsConfigDb configuration;

            configuration.set("RECONFIG_TEST_HOOK_LIBRARY.First",
                              PLUGIN_LIB_DIR "libtestpluginA" PLUGIN_EXT);
            configuration.set("RECONFIG_TEST.First.VALUE", "FirstValue");

            configuration.set("RECONFIG_TEST_HOOK_LIBRARY.Second",
                              PLUGIN_LIB_DIR "libtestpluginA" PLUGIN_EXT);
            configuration.set("RECONFIG_TEST.Second.VALUE", "SecondValue");

            // load them up
            testPlugins.readConfig(configuration);

            PluginIterator plugin(testPlugins);

            // check the first instance
            thePlugin = static_cast<TestPlugin*>(plugin.next(&name));
            CPPUNIT_ASSERT(thePlugin != NULL);
            CPPUNIT_ASSERT(!name.isNull());
            ASSERT_STR_EQUAL("First",name.data());
            thePlugin->pluginName(fullName);
            ASSERT_STR_EQUAL("TestPluginA::First",fullName.data());
            CPPUNIT_ASSERT(thePlugin->getConfiguredValueFor(ValueKey, pluginValue));
            ASSERT_STR_EQUAL("FirstValue",pluginValue.data());

            // check the second instance
            thePlugin = static_cast<TestPlugin*>(plugin.next(&name));
            CPPUNIT_ASSERT(thePlugin != NULL);
            CPPUNIT_ASSERT(!name.isNull());
            ASSERT_STR_EQUAL("Second",name.data());
            thePlugin->pluginName(fullName);
            ASSERT_STR_EQUAL("TestPluginA::Second",fullName.data());
            CPPUNIT_ASSERT(thePlugin->getConfiguredValueFor(ValueKey, pluginValue));
            ASSERT_STR_EQUAL("SecondValue",pluginValue.data());

            // and confirm that is the end
            CPPUNIT_ASSERT(plugin.next(&name) == NULL);
            CPPUNIT_ASSERT(name.isNull());
         }

         {
            // Now create a new configuration that eliminates the First instance
            // and changes the configuration value for the Second instance
            OsConfigDb configuration;
            configuration.set("RECONFIG_TEST_HOOK_LIBRARY.Second",
                              PLUGIN_LIB_DIR "libtestpluginA" PLUGIN_EXT);
            configuration.set("RECONFIG_TEST.Second.VALUE", "NewValue");

            // reconfigure the plugins
            testPlugins.readConfig(configuration);

            PluginIterator plugin(testPlugins);

            // confirm that we still get the Second instance (but we get it first :-)
            thePlugin = static_cast<TestPlugin*>(plugin.next(&name));
            CPPUNIT_ASSERT(thePlugin != NULL);
            CPPUNIT_ASSERT(!name.isNull());
            ASSERT_STR_EQUAL("Second",name.data());
            thePlugin->pluginName(fullName);
            ASSERT_STR_EQUAL("TestPluginA::Second",fullName.data());

            // and check that it has the new configuration value
            CPPUNIT_ASSERT(thePlugin->getConfiguredValueFor(ValueKey, pluginValue));
            ASSERT_STR_EQUAL("NewValue",pluginValue.data());

            // and that it's the only hook configured now
            CPPUNIT_ASSERT(plugin.next(&name) == NULL);
            CPPUNIT_ASSERT(name.isNull());
         }

         {
            // Now create a third configuration that changes the library for the Second instance
            OsConfigDb configuration;
            configuration.set("RECONFIG_TEST_HOOK_LIBRARY.Second",
                              PLUGIN_LIB_DIR "libtestpluginB" PLUGIN_EXT);
            configuration.set("RECONFIG_TEST.Second.VALUE", "ChangedValue");

            // reconfigure the plugins
            testPlugins.readConfig(configuration);

            PluginIterator plugin(testPlugins);

            // confirm that we still get the Second instance (but we get it first :-)
            thePlugin = static_cast<TestPlugin*>(plugin.next(&name));
            CPPUNIT_ASSERT(thePlugin != NULL);
            CPPUNIT_ASSERT(!name.isNull());
            ASSERT_STR_EQUAL("Second",name.data());
            thePlugin->pluginName(fullName);
            ASSERT_STR_EQUAL("TestPluginB::Second",fullName.data());

            // and check that it has the new configuration value
            CPPUNIT_ASSERT(thePlugin->getConfiguredValueFor(ValueKey, pluginValue));
            ASSERT_STR_EQUAL("ChangedValue",pluginValue.data());

            // and that it's the only hook configured now
            CPPUNIT_ASSERT(plugin.next(&name) == NULL);
            CPPUNIT_ASSERT(name.isNull());
         }
      }

   void testNoLibrary()
      {
         OsConfigDb configuration;

         configuration.set("NO_LIB_NOTPASSED_HOOK_LIBRARY.Error", PLUGIN_LIB_DIR "libfoo" PLUGIN_EXT);
         configuration.set("NO_LIB_OTHERPARAM", "DummyValue");

         // configure two instances of the same hook library, with different parameters
         configuration.set("NO_LIB_TEST_HOOK_LIBRARY.1-First",
                           PLUGIN_LIB_DIR "libtestpluginA" PLUGIN_EXT);
         configuration.set("NO_LIB_TEST.1-First.VALUE", "FirstValue");

         configuration.set("NO_LIB_TEST_HOOK_LIBRARY.2-EmptyLib", "");
         configuration.set("NO_LIB_TEST.2-EmptyLib.VALUE", "EmptyLibValue");

         configuration.set("NO_LIB_TEST_HOOK_LIBRARY.3-Third",
                           PLUGIN_LIB_DIR "libtestpluginB" PLUGIN_EXT);
         configuration.set("NO_LIB_TEST.3-Third.VALUE", "ThirdValue");

         // load up the hooks
         PluginHooks testPlugins("getTestPlugin", "NO_LIB_TEST");
         testPlugins.readConfig(configuration);
         PluginIterator plugin(testPlugins);

         UtlString name;
         TestPlugin* thePlugin;

         // get the first instance and check its name
         thePlugin = static_cast<TestPlugin*>(plugin.next(&name));
         CPPUNIT_ASSERT(thePlugin != NULL);
         CPPUNIT_ASSERT(!name.isNull());
         ASSERT_STR_EQUAL("1-First",name.data());
         UtlString fullName;
         thePlugin->pluginName(fullName);
         ASSERT_STR_EQUAL("TestPluginA::1-First",fullName.data());

         // get the second instance and confirm its name
         thePlugin = static_cast<TestPlugin*>(plugin.next(&name));
         CPPUNIT_ASSERT(thePlugin != NULL);
         CPPUNIT_ASSERT(!name.isNull());
         ASSERT_STR_EQUAL("3-Third",name.data());
         thePlugin->pluginName(fullName);
         ASSERT_STR_EQUAL("TestPluginB::3-Third",fullName.data());

         // and make sure that is the end of the iteration
         CPPUNIT_ASSERT(plugin.next(&name) == NULL);
         CPPUNIT_ASSERT(name.isNull());
      }

};

CPPUNIT_TEST_SUITE_REGISTRATION(PluginHooksTest);
