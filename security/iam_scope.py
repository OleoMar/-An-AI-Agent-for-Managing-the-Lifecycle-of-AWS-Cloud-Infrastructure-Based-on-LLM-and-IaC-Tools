"""
security/iam_scope.py

Принцип минимальных привилегий для IAM.

Проблема:
  Если дать агенту AdministratorAccess — он может случайно или
  намеренно сделать что угодно в AWS аккаунте.

Решение:
  Для каждого типа ресурса определяем минимальный набор IAM actions
  который нужен для создания/изменения/удаления именно этого ресурса.
  Агент работает только с теми правами которые нужны для конкретного плана.

Использование:
  from security.iam_scope import get_required_permissions
  permissions = get_required_permissions(plan)
  # → ['s3:CreateBucket', 's3:PutBucketVersioning', ...]
"""

from schemas.plan_schema import Plan, ResourceType


# ── Маппинг типов ресурсов → минимальные IAM actions ─────────────────────────
# Только то что реально нужно Terraform для CRUD операций с каждым типом.

_RESOURCE_PERMISSIONS: dict[ResourceType, list[str]] = {

    ResourceType.S3_BUCKET: [
        "s3:CreateBucket",
        "s3:DeleteBucket",
        "s3:GetBucketLocation",
        "s3:GetBucketVersioning",
        "s3:PutBucketVersioning",
        "s3:PutBucketEncryption",
        "s3:GetEncryptionConfiguration",
        "s3:PutBucketPublicAccessBlock",
        "s3:GetBucketPublicAccessBlock",
        "s3:PutBucketPolicy",
        "s3:GetBucketPolicy",
        "s3:DeleteBucketPolicy",
        "s3:GetBucketWebsite",
        "s3:PutBucketWebsite",
        "s3:DeleteBucketWebsite",
        "s3:GetBucketTagging",
        "s3:PutBucketTagging",
        "s3:GetBucketLogging",
        "s3:PutBucketLogging",
    ],

    ResourceType.EC2_INSTANCE: [
        "ec2:RunInstances",
        "ec2:TerminateInstances",
        "ec2:DescribeInstances",
        "ec2:DescribeInstanceStatus",
        "ec2:StartInstances",
        "ec2:StopInstances",
        "ec2:CreateTags",
        "ec2:DeleteTags",
        "ec2:DescribeTags",
    ],

    ResourceType.VPC: [
        "ec2:CreateVpc",
        "ec2:DeleteVpc",
        "ec2:DescribeVpcs",
        "ec2:ModifyVpcAttribute",
        "ec2:CreateTags",
    ],

    ResourceType.SUBNET: [
        "ec2:CreateSubnet",
        "ec2:DeleteSubnet",
        "ec2:DescribeSubnets",
        "ec2:ModifySubnetAttribute",
        "ec2:CreateTags",
    ],

    ResourceType.SECURITY_GROUP: [
        "ec2:CreateSecurityGroup",
        "ec2:DeleteSecurityGroup",
        "ec2:DescribeSecurityGroups",
        "ec2:AuthorizeSecurityGroupIngress",
        "ec2:AuthorizeSecurityGroupEgress",
        "ec2:RevokeSecurityGroupIngress",
        "ec2:RevokeSecurityGroupEgress",
        "ec2:CreateTags",
    ],

    ResourceType.INTERNET_GATEWAY: [
        "ec2:CreateInternetGateway",
        "ec2:DeleteInternetGateway",
        "ec2:AttachInternetGateway",
        "ec2:DetachInternetGateway",
        "ec2:DescribeInternetGateways",
        "ec2:CreateTags",
    ],

    ResourceType.RDS_INSTANCE: [
        "rds:CreateDBInstance",
        "rds:DeleteDBInstance",
        "rds:DescribeDBInstances",
        "rds:ModifyDBInstance",
        "rds:CreateDBSubnetGroup",
        "rds:DeleteDBSubnetGroup",
        "rds:DescribeDBSubnetGroups",
        "rds:AddTagsToResource",
        "rds:ListTagsForResource",
        "rds:CreateDBSnapshot",
    ],

    ResourceType.DYNAMODB_TABLE: [
        "dynamodb:CreateTable",
        "dynamodb:DeleteTable",
        "dynamodb:DescribeTable",
        "dynamodb:UpdateTable",
        "dynamodb:TagResource",
        "dynamodb:UntagResource",
        "dynamodb:ListTagsOfResource",
    ],

    ResourceType.IAM_ROLE: [
        "iam:CreateRole",
        "iam:DeleteRole",
        "iam:GetRole",
        "iam:UpdateRole",
        "iam:PassRole",
        "iam:AttachRolePolicy",
        "iam:DetachRolePolicy",
        "iam:ListAttachedRolePolicies",
        "iam:TagRole",
        "iam:UntagRole",
    ],

    ResourceType.IAM_POLICY: [
        "iam:CreatePolicy",
        "iam:DeletePolicy",
        "iam:GetPolicy",
        "iam:GetPolicyVersion",
        "iam:CreatePolicyVersion",
        "iam:DeletePolicyVersion",
        "iam:ListPolicyVersions",
        "iam:TagPolicy",
    ],

    ResourceType.LAMBDA_FUNCTION: [
        "lambda:CreateFunction",
        "lambda:DeleteFunction",
        "lambda:GetFunction",
        "lambda:UpdateFunctionCode",
        "lambda:UpdateFunctionConfiguration",
        "lambda:AddPermission",
        "lambda:RemovePermission",
        "lambda:TagResource",
        "lambda:UntagResource",
        "lambda:CreateEventSourceMapping",
        "lambda:DeleteEventSourceMapping",
        "lambda:GetEventSourceMapping",
    ],

    ResourceType.SQS_QUEUE: [
        "sqs:CreateQueue",
        "sqs:DeleteQueue",
        "sqs:GetQueueAttributes",
        "sqs:SetQueueAttributes",
        "sqs:TagQueue",
        "sqs:UntagQueue",
        "sqs:ListQueueTags",
    ],

    ResourceType.SNS_TOPIC: [
        "sns:CreateTopic",
        "sns:DeleteTopic",
        "sns:GetTopicAttributes",
        "sns:SetTopicAttributes",
        "sns:TagResource",
        "sns:UntagResource",
    ],

    ResourceType.ALB: [
        "elasticloadbalancing:CreateLoadBalancer",
        "elasticloadbalancing:DeleteLoadBalancer",
        "elasticloadbalancing:DescribeLoadBalancers",
        "elasticloadbalancing:ModifyLoadBalancerAttributes",
        "elasticloadbalancing:AddTags",
        "elasticloadbalancing:RemoveTags",
    ],

    ResourceType.ALB_TARGET_GROUP: [
        "elasticloadbalancing:CreateTargetGroup",
        "elasticloadbalancing:DeleteTargetGroup",
        "elasticloadbalancing:DescribeTargetGroups",
        "elasticloadbalancing:ModifyTargetGroup",
        "elasticloadbalancing:AddTags",
    ],

    ResourceType.EBS_VOLUME: [
        "ec2:CreateVolume",
        "ec2:DeleteVolume",
        "ec2:DescribeVolumes",
        "ec2:AttachVolume",
        "ec2:DetachVolume",
        "ec2:ModifyVolume",
        "ec2:CreateTags",
    ],
}

# Базовые права которые нужны Terraform всегда
_BASE_PERMISSIONS = [
    "sts:GetCallerIdentity",   # terraform init
    "s3:GetObject",            # скачивание провайдера (если state в S3)
]


def get_required_permissions(plan: Plan) -> list[str]:
    """
    Возвращает минимальный список IAM actions для всех ресурсов плана.

    Параметры:
        plan — валидный объект Plan

    Возвращает:
        Отсортированный список уникальных IAM actions.
    """
    permissions: set[str] = set(_BASE_PERMISSIONS)

    for resource in plan.resources:
        resource_perms = _RESOURCE_PERMISSIONS.get(resource.type, [])
        permissions.update(resource_perms)

        if not resource_perms:
            # Тип есть в схеме но нет в маппинге — логируем предупреждение
            import logging
            logging.getLogger(__name__).warning(
                "[iam_scope] нет маппинга для %s — добавь в _RESOURCE_PERMISSIONS",
                resource.type.value,
            )

    return sorted(permissions)


def format_policy_document(plan: Plan) -> dict:
    """
    Генерирует IAM Policy Document в формате AWS JSON.
    Можно передать напрямую в boto3 или использовать в Terraform.

    Возвращает словарь готовый для json.dumps().
    """
    permissions = get_required_permissions(plan)

    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "AgentMinimalPermissions",
                "Effect": "Allow",
                "Action": permissions,
                "Resource": "*",
                "Condition": {
                    "StringEquals": {
                        "aws:RequestedRegion": plan.aws_region or "us-east-1"
                    }
                }
            }
        ]
    }
