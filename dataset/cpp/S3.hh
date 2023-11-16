#pragma once

#include <map>
#include <vector>

#include <boost/property_tree/ptree.hpp>

#include <elle/Buffer.hh>
#include <elle/Printable.hh>
#include <elle/attribute.hh>
#include <elle/service/aws/CanonicalRequest.hh>
#include <elle/service/aws/Credentials.hh>
#include <elle/service/aws/Exceptions.hh>
#include <elle/service/aws/StringToSign.hh>

namespace elle
{
  namespace service
  {
    namespace aws
    {
      struct URL
      {
        std::string scheme;
        std::string domain;
        std::string path;

        std::string
        join() const;
      };

      class S3
        : public elle::Printable
      {
        /*------.
        | Types |
        `------*/
      public:
        using ProgressCallback = std::function<void (int)>;
        using FileSize = uint64_t;
        using List = std::vector<std::pair<std::string, FileSize>>;

        enum class StorageClass
        {
          Standard,
          StandardIA,
          ReducedRedundancy,

          Default, /// Do not set the x-amz-storage-class header.
        };

        /// Conversion to the syntax expected in headers, except for
        /// `Default` (mapped to `"default"`), which should not be
        /// used in x-amz-storage-class.
        friend std::string to_string(StorageClass c);

        /*-------------.
        | Construction |
        `-------------*/
      public:
        /// Create a new S3 handler.
        /// This requires a bucket name, remote folder and set of
        /// credentials.
        /// Use of this should be optimised based on S3 pricing:
        /// http://aws.amazon.com/s3/pricing/
        S3(Credentials const& credentials);

        using QueryCredentials = std::function<auto (bool) -> Credentials>;
        /// Version taking a function able to refresh credentials.
        /// Bool argument is true on the first call, false on call caused by
        /// expiration of current cached creds.
        S3(QueryCredentials query_credentials);

        /*-----------.
        | Operations |
        `-----------*/
      public:
        /// Put a block (as a file) into the remote folder.
        /// This function does a single PUT call, so block size should be
        /// something reasonable for an HTTP request.
        /// @return the object ETag.
        std::string
        put_object(
          elle::ConstWeakBuffer const& object,
          std::string const& object_name,
          RequestQuery const& query = {},
          StorageClass storage_class = StorageClass::Default,
          boost::optional<ProgressCallback> const& progress_callback = {});

        /// A list of all files names and their respective sizes inside
        /// the remote folder.
        ///
        /// This is limited to 1000 results but a starting offset (marker) can
        /// be used if more are required. The files are listed in alphabetical
        /// order.
        List
        list_remote_folder(std::string const& marker = {});
        /// List the full folder content.
        List
        list_remote_folder_full();
        /// Fetch an object from the remote folder.
        /// The fetch is done in a single GET.
        elle::Buffer
        get_object(std::string const& object_name,
                   RequestHeaders headers = RequestHeaders());
        /// Fetch one chunk of an object.
        elle::Buffer
        get_object_chunk(std::string const& object_name,
                         FileSize offset, FileSize size);
        /// Delete an object in the remote folder.
        /// The folder itself can be deleted only once it is empty. This can be
        /// done by setting the object_name to an empty string.
        void
        delete_object(std::string const& object_name,
                      RequestQuery const& query = RequestQuery());
        /// Delete folder and all its content
        void
        delete_folder();

        /// Initialize multipart upload for given object
        /// @return an upload key needed by further operations
        std::string
        multipart_initialize(
          std::string const& object_name,
          std::string const& mime_type = "binary/octet-stream",
          StorageClass storage_class = StorageClass::Default);

        using MultiPartChunk = std::pair<int, std::string>;
        /// Upload one part of a multipart upload
        /// @return id for chunk information that needs to be passed to
        /// multipart_finalize
        std::string
        multipart_upload(
          std::string const& object_name,
          std::string const& upload_key,
          elle::ConstWeakBuffer const& object,
          int chunk,
          boost::optional<ProgressCallback> const& progress_callback = {});

        /// Finalize a multipart by joining all parts together
        void
        multipart_finalize(std::string const& object_name,
                           std::string const& upload_key,
                           std::vector<MultiPartChunk> const& chunks);

        /// Abort a multipart transfer, delete all chunks
        void
        multipart_abort(std::string const& object_name,
                        std::string const& upload_key);

        std::vector<MultiPartChunk>
        multipart_list(std::string const& object_name,
                       std::string const& upload_key);

        /*-----------.
        | Attributes |
        `-----------*/
      public:
        /// Get the host including the http scheme to connect to depending on
        /// the credentials.
        virtual
        URL
        hostname(Credentials const& credentials,
                 boost::optional<std::string> override_host = {}) const;
        ELLE_ATTRIBUTE(Credentials, credentials);
        ELLE_ATTRIBUTE(QueryCredentials, query_credentials);

        /*--------.
        | Helpers |
        `--------*/
      private:
        enum class RequestKind
        {
          control, // Expected a small amount of data in/out
          data // expected to transfer some amount of data in our out
        };

        std::string
        _md5_digest(elle::ConstWeakBuffer const& buffer);

        std::string
        _sha256_hexdigest(elle::ConstWeakBuffer const& buffer);

        std::string
        _amz_date(RequestTime const& request_time);

        std::vector<std::string>
        _signed_headers(RequestHeaders const& headers);

        StringToSign
        _make_string_to_sign(RequestTime const& request_time,
                             std::string const& canonical_request_sha256);

        List
        _parse_list_xml(std::istream& stream);

        elle::reactor::http::Request::Configuration
        _initialize_request(RequestKind kind,
                            RequestTime request_time,
                            CanonicalRequest const& canonical_request,
                            const RequestHeaders& initial_headers,
                            Duration timeout);

        /// Check return code and throw appropriate exception if error
        /// ELLE_WARN the request response in case of error
        /// Dumps it if dump_response is true, which eats the stream content.
        void
        _check_request_status(elle::reactor::http::Request& request,
                              std::string const& operation);

        /// Build and emit request, retries in case of credentials expiry kind
        /// is used to switch between global duration timeout and stall timeout.
        std::unique_ptr<elle::reactor::http::Request>
        _build_send_request(
          RequestKind kind,
          std::string const& url,
          std::string const& operation, // Used by exception message/log only.
          elle::reactor::http::Method method,
          RequestQuery const& query = RequestQuery(),
          RequestHeaders const& extra_headers = RequestHeaders(),
          std::string const& content_type = "application/json",
          elle::ConstWeakBuffer const& payload = elle::ConstWeakBuffer(),
          DurationOpt timeout = {},
          boost::optional<ProgressCallback> const& progress_callback = {});

        /*----------.
        | Printable |
        `----------*/
      public:
        virtual
        void
        print(std::ostream& stream) const;
        // Callback invoked on all errors, transient or not
        using ErrorCallback =
          std::function<void(AWSException const&, bool will_retry)>;
        ELLE_ATTRIBUTE_RW(ErrorCallback, on_error);
      };

      std::string
      uri_encode(std::string const& input, bool encodeSlash);
    }
  }
}
