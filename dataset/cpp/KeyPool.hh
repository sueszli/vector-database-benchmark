#ifndef ELLE_CRYPTOGRAPHY_RSA_KEYPOOL_HH
# define ELLE_CRYPTOGRAPHY_RSA_KEYPOOL_HH

# include <condition_variable>
# include <thread>
# include <vector>

# include <elle/cryptography/rsa/KeyPair.hh>
# include <elle/ProducerPool.hh>

namespace elle
{
  namespace cryptography
  {
    namespace rsa
    {
      class KeyPool
        : public elle::ProducerPool<KeyPair>
      {
      public:
        KeyPool(int key_size, int max_pool_size, int thread_count = 1)
        : ProducerPool<KeyPair>(
            [key_size] { return keypair::generate(key_size);},
            max_pool_size,
            thread_count)
        {}
      };
    }
  }
}

#endif
