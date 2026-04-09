import datetime
import random

from typing import Annotated

from pydantic import BaseModel, EmailStr, computed_field, model_validator, Field
from typing_extensions import Self

from agentdojo.default_suites.v1.tools.dynamic_web_client import WebDatabase, upload_file_to_web
from agentdojo.default_suites.v1.tools.file_client import Filesystem, check_node_existence
from agentdojo.default_suites.v1.tools.types import CalendarEvent, CloudDriveFileID, Email, EmailID, EmailStatus, WebFileID, File
from agentdojo.functions_runtime import Depends
from agentdojo.default_suites.v1.tools.dynamic_email_client import EmailDatabase, send_receive_email
from agentdojo.default_suites.v1.tools.banking_client import BankAccount, Transaction, next_id, get_iban

from agentdojo.default_suites.v1.tools.tool_registry import register_tool


class DynamicBankAccount(BankAccount):
    password: str = Field(default="")
    verification: bool = Field(default=False)
    verification_stack: dict[str, tuple] = Field(default_factory=dict)

class BankDatabase(BaseModel):
    account_list: list[DynamicBankAccount]
    current_account: str

def get_bank_account_instance(
    bank_database: Annotated[BankDatabase, Depends("bank_database")], account_number: str
) -> DynamicBankAccount | str:
    """
    Get the bank account instance of the given account number.

    :param account_number: Account number to get the bank account instance
    """
    for bank_account in bank_database.account_list:
        if bank_account.iban == account_number:
            return bank_account

    return "Bank account not found."

def send_otp_to_inbox(
    bank_database: Annotated[BankDatabase, Depends("bank_database")], email_database: Annotated[EmailDatabase, Depends("email_database")], filesystem: Annotated[Filesystem, Depends("filesystem")], web_database: Annotated[WebDatabase, Depends("web_database")], sender: str, subject: str, body: str
) -> str:
    """
    Sends an OTP to the linked email inbox.
    """
    otp = ''.join(random.choices('0123456789', k=6))

    current_account = get_bank_account_instance(bank_database, account_number=bank_database.current_account)
    if not isinstance(current_account, DynamicBankAccount):
        return "No current bank account found."
    current_account_email = email_database.current_account
    assert isinstance(current_account_email, str), "No current email account found."
    send_receive_email(email_database=email_database, filesystem=filesystem, web_database=web_database, sender=sender, subject=subject, body=body+otp, recipients=[current_account_email])

    # return "The OTP has been sent to your email inbox."
    return otp
