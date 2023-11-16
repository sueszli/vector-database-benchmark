#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <QString>
#include <login_dto.hpp>
#include "authentication_controller.hpp"
#include "i_authentication_service.hpp"


using namespace testing;
using ::testing::ReturnRef;
using namespace adapters::controllers;
using namespace adapters;
using namespace domain::value_objects;
using namespace application;

namespace tests::adapters
{

class AuthenticationServiceMock : public application::IAuthenticationService
{
public:
    MOCK_METHOD(void, loginUser, (const LoginModel&), (override));
    MOCK_METHOD(void, tryAutomaticLogin, (), (override));
    MOCK_METHOD(void, logoutUser, (), (override));
    MOCK_METHOD(void, registerUser,
                (const domain::value_objects::RegisterModel&), (override));
    MOCK_METHOD(void, processAuthenticationResult, (const QString&, ErrorCode),
                (override));
    MOCK_METHOD(void, processRegistrationResult, (ErrorCode), (override));
    MOCK_METHOD(void, checkIfEmailConfirmed, (const QString&), (override));
};

struct AnAuthenticationController : public ::testing::Test
{
    void SetUp() override
    {
        authController =
            std::make_unique<AuthenticationController>(&authServiceMock);
    }

    AuthenticationServiceMock authServiceMock;
    std::unique_ptr<AuthenticationController> authController;
};

TEST_F(AnAuthenticationController, SucceedsLogingAUserIn)
{
    // Arrange
    QString email = "SomeEmail@librum.com";
    QString password = "SomePassword12345";


    // Expect
    EXPECT_CALL(authServiceMock, loginUser(_)).Times(1);


    // Act
    authController->loginUser(email, password, false);
}

TEST_F(AnAuthenticationController, SucceedsRegisteringAUser)
{
    // Arrange
    QString firstName = "Kai";
    QString lastName = "Doe";
    QString email = "SomeEmail@librum.com";
    QString password = "SomePassword12345";
    bool keepUpdated = true;


    // Expect
    EXPECT_CALL(authServiceMock, registerUser(_)).Times(1);

    // Act
    authController->registerUser(firstName, lastName, email, password,
                                 keepUpdated);
}

TEST_F(AnAuthenticationController, SucceedsLogingOutAUser)
{
    // Expect
    EXPECT_CALL(authServiceMock, logoutUser()).Times(1);

    // Act
    authController->logoutUser();
}

}  // namespace tests::adapters