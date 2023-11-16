#pragma once

#include <elle/Error.hh>
#include <elle/optional.hh>
#include <elle/reactor/Channel.hh>
#include <elle/reactor/Thread.hh>

namespace elle
{
  namespace reactor
  {
    template <typename T>
    using yielder = std::function<void (T)>;

    /// Create an iterable object over a function. This is powerful in
    /// asynchronous environments when one needs to wait for at least n values
    /// (see example).
    ///
    /// Since the results are put in a Channel, you can get all the
    /// benefits of the Channel (mostly waiting for result).
    ///
    /// \code{.cc}
    ///
    /// auto f = [&] (elle::reactor::yielder<int>::type const& yield)
    ///   {
    ///     elle::With<elle::reactor::Scope>() << [&](elle::reactor::Scope &s)
    ///     {
    ///       // Lets consider `sources`, a list of sources you want to check to
    ///       // make sure some information is relevant.
    ///       for (auto const& source: sources)
    ///       {
    ///         s.run_background([&source]
    ///                          {
    ///                            reactor::http::Request r(source);
    ///                            r.finalize();
    ///                            if (r.ok())
    ///                              yield(source);
    ///                          });
    ///       }
    ///       scope.wait();
    ///     };
    ///   };
    /// for (auto const& response: elle::reactor::generator(f))
    ///   std::cout << response;
    /// // Result: Only the valid sources.
    ///
    /// \endcode
    template <typename T>
    struct Generator
    {
    /*----.
    | End |
    `----*/
    public:
      class End
        : public elle::Error
      {
      public:
        End(Generator const& g);
      };

    /*-------------.
    | Construction |
    `-------------*/
    public:
      /// A consumer of the generated values.
      using yielder = elle::reactor::yielder<T>;

      /// Create a generator on a driver.
      ///
      /// The signature of the Driver must be auto `(yielder const&) -> void`.
      template <typename Driver>
      Generator(Driver driver);
      Generator(Generator&& b);
      ~Generator();

    /*--------.
    | Content |
    `--------*/
    public:
      /// Get the next result element of the Generator.
      ///
      /// If it's not yet available, wait until it is.
      /// If there is not more element, throw a End exception.
      ///
      /// \returns The next element.
      T
      next();

    /*---------.
    | Iterator |
    `---------*/
    public:
      /// Iterator used by the Generator.
      struct iterator
        : public std::iterator<std::input_iterator_tag, T>
      {
        using reference = T;
        iterator();
        iterator(Generator<T>& generator);
        /// Compare iterators.
        bool
        operator !=(iterator const& other) const;
        /// Compare iterators.
        bool
        operator ==(iterator const& other) const;
        /// Advance the iterator.
        ///
        /// @return This.
        iterator&
        operator ++();
        /// Advance the iterator.
        ///
        /// @return
        void
        operator ++(int);
        /// Return the element previous to current.
        T
        operator *() const;
        ELLE_ATTRIBUTE(Generator<T>*, generator)
        ELLE_ATTRIBUTE(boost::optional<T>, value, mutable);
        ELLE_ATTRIBUTE(bool, fetch, mutable);
      };
      using const_iterator = iterator;

      /// An iterator to the beginning of the Generator.
      iterator
      begin();
      /// An iterator to the end of the Generator.
      iterator
      end();
      /// An iterator to the beginning of the Generator.
      const_iterator
      begin() const;
      /// An iterator to the end of the Generator.
      const_iterator
      end() const;
      ELLE_ATTRIBUTE(reactor::Channel<boost::optional<T>>, results);
      ELLE_ATTRIBUTE(std::exception_ptr, exception);
      ELLE_ATTRIBUTE(reactor::Thread::unique_ptr, thread);
    };

    /// Construct a generator from a Driver.
    template <typename T>
    Generator<T>
    generator(std::function<void (yielder<T> const&)> const& driver);
  }
}

# include <elle/reactor/Generator.hxx>
