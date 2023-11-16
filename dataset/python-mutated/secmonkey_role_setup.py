"""
SecurityMonkey AWS role provisioning script
Grab credentials from ~/.boto (or other standard credentials sources).
Optionally accept "profile_name" as CLI parameter.
"""
import sys, json
import urllib
import boto
secmonkey_arn = 'arn:aws:iam::<awsaccountnumber>:role/SecurityMonkeyInstanceProfile'
trust_relationship = '\n{\n  "Version": "2008-10-17",\n  "Statement": [\n    {\n      "Sid": "",\n      "Effect": "Allow",\n      "Principal": {\n        "AWS": "%s"\n      },\n      "Action": "sts:AssumeRole"\n    }\n  ]\n}\n'
role_name = 'SecurityMonkey'
role_policy_name = 'SecurityMonkeyPolicy'
policy = '\n{\n  "Statement": [\n    {\n      "Action": [\n          "acm:describecertificate",\n           "acm:listcertificates",\n           "cloudtrail:describetrails",\n           "cloudtrail:gettrailstatus",\n           "config:describeconfigrules",\n           "config:describeconfigurationrecorders",\n           "directconnect:describeconnections",\n           "ec2:describeaddresses",\n           "ec2:describedhcpoptions",\n           "ec2:describeflowlogs",\n           "ec2:describeimages",\n           "ec2:describeimageattribute",\n           "ec2:describeinstances",\n           "ec2:describeinternetgateways",\n           "ec2:describekeypairs",\n           "ec2:describenatgateways",\n           "ec2:describenetworkacls",\n           "ec2:describenetworkinterfaces",\n           "ec2:describeregions",\n           "ec2:describeroutetables",\n           "ec2:describesecuritygroups",\n           "ec2:describesnapshots",\n           "ec2:describesnapshotattribute",\n           "ec2:describesubnets",\n           "ec2:describetags",\n           "ec2:describevolumes",\n           "ec2:describevpcendpoints",\n           "ec2:describevpcpeeringconnections",\n           "ec2:describevpcs",\n           "ec2:describevpnconnections",\n           "ec2:describevpngateways",\n           "elasticloadbalancing:describeloadbalancerattributes",\n           "elasticloadbalancing:describeloadbalancerpolicies",\n           "elasticloadbalancing:describeloadbalancers",\n           "elasticloadbalancing:describelisteners",\n           "elasticloadbalancing:describerules",\n           "elasticloadbalancing:describesslpolicies",\n           "elasticloadbalancing:describetags",\n           "elasticloadbalancing:describetargetgroups",\n           "elasticloadbalancing:describetargetgroupattributes",\n           "elasticloadbalancing:describetargethealth",\n           "es:describeelasticsearchdomainconfig",\n           "es:listdomainnames",\n           "iam:getaccesskeylastused",\n           "iam:getgroup",\n           "iam:getgrouppolicy",\n           "iam:getloginprofile",\n           "iam:getpolicyversion",\n           "iam:getrole",\n           "iam:getrolepolicy",\n           "iam:getservercertificate",\n           "iam:getuser",\n           "iam:getuserpolicy",\n           "iam:listaccesskeys",\n           "iam:listattachedgrouppolicies",\n           "iam:listattachedrolepolicies",\n           "iam:listattacheduserpolicies",\n           "iam:listentitiesforpolicy",\n           "iam:listgrouppolicies",\n           "iam:listgroups",\n           "iam:listinstanceprofilesforrole",\n           "iam:listmfadevices",\n           "iam:listpolicies",\n           "iam:listrolepolicies",\n           "iam:listroles",\n           "iam:listroletags",\n           "iam:listsamlproviders",\n           "iam:listservercertificates",\n           "iam:listsigningcertificates",\n           "iam:listuserpolicies",\n           "iam:listusers",\n           "kms:describekey",\n           "kms:getkeypolicy",\n           "kms:getkeyrotationstatus",\n           "kms:listaliases",\n           "kms:listgrants",\n           "kms:listkeypolicies",\n           "kms:listkeys",\n           "lambda:listfunctions",\n           "lambda:getfunctionconfiguration",\n           "lambda:getpolicy",\n           "lambda:listaliases",\n           "lambda:listeventsourcemappings",\n           "lambda:listtags",\n           "lambda:listversionsbyfunction",\n           "lambda:listfunctions",\n           "rds:describedbclusters",\n           "rds:describedbclustersnapshots",\n           "rds:describedbinstances",\n           "rds:describedbsecuritygroups",\n           "rds:describedbsnapshots",\n           "rds:describedbsnapshotattributes",\n           "rds:describedbsubnetgroups",\n           "redshift:describeclusters",\n           "route53:listhostedzones",\n           "route53:listresourcerecordsets",\n           "route53domains:listdomains",\n           "route53domains:getdomaindetail",\n           "s3:getbucketacl",\n           "s3:getbucketlocation",\n           "s3:getbucketlogging",\n           "s3:getbucketpolicy",\n           "s3:getbuckettagging",\n           "s3:getbucketversioning",\n           "s3:getlifecycleconfiguration",\n           "s3:listallmybuckets",\n           "ses:getidentityverificationattributes",\n           "ses:listidentities",\n           "ses:listverifiedemailaddresses",\n           "ses:sendemail",\n           "sns:gettopicattributes",\n           "sns:listsubscriptionsbytopic",\n           "sns:listtopics",\n           "sqs:getqueueattributes",\n           "sqs:listqueues",\n           "sqs:listqueuetags", \n           "sqs:listdeadlettersourcequeues"\n      ],\n      "Effect": "Allow",\n      "Resource": "*"\n    }\n  ]\n}\n'

def main(profile=None):
    if False:
        while True:
            i = 10
    assume_policy = json.dumps(json.loads(trust_relationship % secmonkey_arn))
    security_policy = json.dumps(json.loads(policy))
    (role_exist, current_policy) = (False, '')
    try:
        iam = boto.connect_iam(profile_name=profile)
    except boto.exception.NoAuthHandlerFound:
        sys.exit('Authentication failed, please check your credentials under ~/.boto')
    rlist = iam.list_roles()
    for r in rlist['list_roles_response']['list_roles_result']['roles']:
        if r['role_name'] == role_name:
            role_exist = True
            current_policy = json.loads(urllib.unquote(r['assume_role_policy_document']))
            for p in current_policy['Statement']:
                if p['Action'] == 'sts:AssumeRole':
                    if secmonkey_arn in p['Principal']['AWS']:
                        sys.exit('Role "%s" already configured, not touching it.' % role_name)
                    else:
                        new_policy = [secmonkey_arn]
                        new_policy.extend(p['Principal']['AWS'])
                        p['Principal']['AWS'] = new_policy
            assume_policy = json.dumps(current_policy)
    if not role_exist:
        role = iam.create_role(role_name, assume_policy)
    else:
        role = iam.update_assume_role_policy(role_name, assume_policy)
    iam.put_role_policy(role_name, role_policy_name, security_policy)
    print('Added role "%s", linked to ARN "%s".' % (role_name, secmonkey_arn))
if __name__ == '__main__':
    profile = None
    if len(sys.argv) >= 2:
        profile = sys.argv[1]
    main(profile)