#include <elle/log/CompositeLogger.hh>

#include <boost/algorithm/cxx11/any_of.hpp>

#include <elle/unreachable.hh>

namespace elle
{
  namespace log
  {
    CompositeLogger::CompositeLogger(Loggers l)
      : Super{"LOG"}
      , _loggers{std::move(l)}
    {}

    CompositeLogger::CompositeLogger()
      : Super{"LOG"}
    {}

    void
    CompositeLogger::_message(Message const& msg)
    {
      // We bounce to `Logger::message` and not `Logger::_message` so
      // that each child logger can have its own settings (in
      // particular its log_level).
      //
      // We must reproduce Send's behavior regarding indent and
      // categories.
      for (auto& l: this->_loggers)
      {
        l->indentation() = msg.indentation + 1;
        l->component_level(msg.component); // for max size computation
        l->message(msg);
      }
    }

    bool
    CompositeLogger::_component_is_active(std::string const& name, Level level)
    {
      return boost::algorithm::any_of(this->_loggers,
                                      [&name, level](auto& l)
                                      {
                                        return l->component_is_active(name, level);
                                      });
    }
  }
}
