import datetime
from typing import Annotated

from pydantic import BaseModel, EmailStr, computed_field, model_validator
from typing_extensions import Self

from agentdojo.default_suites.v1.tools.dynamic_web_client import WebDatabase, upload_file_to_web
from agentdojo.default_suites.v1.tools.file_client import Filesystem, check_node_existence
from agentdojo.default_suites.v1.tools.types import CalendarEvent, CloudDriveFileID, Email, EmailID, EmailStatus, WebFileID, File
from agentdojo.functions_runtime import Depends
from agentdojo.default_suites.v1.tools.email_client import EmailContact, Inbox

from agentdojo.default_suites.v1.tools.tool_registry import register_tool


class DynamicInbox(Inbox):
    def receive_email(
        self,
        sender: EmailStr,
        subject: str,
        body: str,
        attachments: list[CloudDriveFileID | CalendarEvent] | None = None,
        cc: list[EmailStr] | None = None,
        bcc: list[EmailStr] | None = None,
    ) -> Email:
        if cc is None:
            cc = []
        if bcc is None:
            bcc = []
        if attachments is None:
            attachments = []
        new_email = Email(
            id_=self._get_next_id(),
            sender=sender,
            body=body,
            subject=subject,
            recipients=[self.account_email],
            status=EmailStatus.received,
            timestamp=datetime.datetime.now(),
            cc=cc,
            bcc=bcc,
            attachments=attachments,
            read=False,
        )
        self.emails[new_email.id_] = new_email
        return new_email

    def send_email(
        self,
        recipients: list[EmailStr],
        subject: str,
        body: str,
        attachments: list[CloudDriveFileID | CalendarEvent] | None = None,
        cc: list[EmailStr] | None = None,
        bcc: list[EmailStr] | None = None,
    ) -> Email:
        if cc is None:
            cc = []
        if bcc is None:
            bcc = []
        if attachments is None:
            attachments = []
        new_email = Email(
            id_=self._get_next_id(),
            sender=self.account_email,
            body=body,
            subject=subject,
            recipients=recipients,
            status=EmailStatus.sent,
            timestamp=datetime.datetime.now(),
            cc=cc,
            bcc=bcc,
            attachments=attachments,
            read=True,
        )
        self.emails[new_email.id_] = new_email
        return new_email

class EmailDatabase(BaseModel):
    inbox_list: list[DynamicInbox]
    current_account: EmailStr | None = None
    
    def get_inbox_by_email(self, email_address: EmailStr) -> DynamicInbox | None:
        for inbox in self.inbox_list:
            if inbox.account_email == email_address:
                return inbox
        return None
    
    def add_new_address(self, email: EmailStr) -> DynamicInbox:
        new_inbox = DynamicInbox(account_email=email, initial_emails=[])
        self.inbox_list.append(new_inbox)
        return new_inbox


def get_email_account_instance(
    email_database: Annotated[EmailDatabase, Depends("email_database")], email_address: EmailStr | None
) -> DynamicInbox | None:
    """
    Get the email account instance of the given email address.

    :param email_address: Email address to get the email account instance
    """
    if email_address is not None:
        cur_inbox = email_database.get_inbox_by_email(email_address)
        if cur_inbox is not None:
            return cur_inbox
        else:
            return email_database.add_new_address(email_address)
    else:
        return None


@register_tool("send_receive_email")
def send_receive_email(
    email_database: Annotated[EmailDatabase, Depends("email_database")],
    filesystem: Annotated[Filesystem, Depends("filesystem")],
    web_database: Annotated[WebDatabase, Depends("web_database")],
    sender: EmailStr,
    subject: str,
    body: str,
    recipients: list[EmailStr] | EmailStr,
    reservation: str | None = None,
    time: str | None = None,
    size: str | None = None,
    attachments: list[dict] | None = None,
    cc: list[EmailStr] | None = None,
    bcc: list[EmailStr] | None = None,
) -> Email | str:
    """Receives an email with the given `body` to the given `address`. Returns a dictionary with the email details.

    :param sender: The email address of the sender.
    :param subject: The subject of the email.
    :param body: The body of the email.
    :param attachments: The list of attachments to include in the email. If `null`, no attachments are included.
    If the attachment has as "type" "file", then it is expected to have a field "file_path", with the path of file in the local file system. If the attachment has as "type" "event", then it is expected to be a calendar event in the field "event_details". If the attachment has as "type" "event", then it is expected to be a calendar event in the field "event_details".
    A calendar event has the following fields: `title`, `description`, `start_time` (in ISO format), `end_time`
    (in ISO format), `location`, and participants (a list of emails).
    :param recipient:  The email address of the recipient.
    :param cc: The list of email addresses to include in the CC field. If `null`, no email addresses are included.
    :param bcc: The list of email addresses to include in the BCC field. If `null`, no email addresses are included.
    """
    if attachments is not None:
        parsed_attachments = []
        for attachment in attachments:
            if not isinstance(attachment, dict):
                raise ValueError("Attachments must be dictionaries.")
            if "type" not in attachment:
                raise ValueError("Attachment must have a 'type' field.")
            if attachment["type"] == "file" or "file_path" in attachment or "file_id" in attachment:
                if ("file_path" not in attachment) and ("file_id" not in attachment):
                    raise ValueError("Attachment of type 'file' must have a 'file_path' or 'file_id' field.")
                if "file_path" in attachment:
                    file_id = upload_file_to_web(web_database=web_database, filesystem=filesystem, local_file_path=attachment["file_path"])
                    parsed_attachments.append(file_id)
                elif "file_id" in attachment:
                    file_id = attachment["file_id"]
                    parsed_attachments.append(file_id)
            else:
                if "event_details" not in attachment:
                    raise ValueError("Attachment of type 'event' must have an 'event_details' field.")
                parsed_attachments.append(CalendarEvent(**attachment["event_details"]))
    else:
        parsed_attachments = None
    
    if reservation is not None:
        assert time is not None and size is not None, "Expected Time and Party Size must be provided for reservation."
        body = body.replace("@@@time@@@", time).replace("@@@size@@@", size)

    sender_inbox = get_email_account_instance(email_database, sender)
    recipient_inboxs = []
    msg = ""

    if isinstance(recipients, str):
        recipients = [recipients]

    for recipient in recipients:
        if recipient is not None:
            cur_inbox = get_email_account_instance(email_database, recipient)
            if cur_inbox is not None:
                recipient_inboxs.append(cur_inbox)
            else:
                recipient_inboxs.append(email_database.add_new_address(recipient))
    
    if (sender_inbox is None) and (len(recipient_inboxs) == 0):
        raise ValueError("Email account invalid.")

    if sender_inbox is not None:
        msg = sender_inbox.send_email(recipients=recipients, subject=subject, body=body, attachments=parsed_attachments, cc=cc, bcc=bcc)
    
    for recipient_inbox in recipient_inboxs:
        msg = recipient_inbox.receive_email(sender, subject, body, parsed_attachments, cc, bcc)

    return msg


@register_tool("send_receive_email_with_login")
def send_receive_email_with_login(
    email_database: Annotated[EmailDatabase, Depends("email_database")],
    filesystem: Annotated[Filesystem, Depends("filesystem")],
    web_database: Annotated[WebDatabase, Depends("web_database")],
    username: str,
    password: str,
    username_groundtruth: str,
    password_groundtruth: str,
    sender: EmailStr,
    subject: str,
    body: str,
    recipients: list[EmailStr] | EmailStr,
    return_str: str | None = None,
    attachments: list[dict] | None = None,
    cc: list[EmailStr] | None = None,
    bcc: list[EmailStr] | None = None,
) -> Email | str:
    """Receives an email with the given `body` to the given `address`. Returns a dictionary with the email details.

    :param sender: The email address of the sender.
    :param subject: The subject of the email.
    :param body: The body of the email.
    :param attachments: The list of attachments to include in the email. If `null`, no attachments are included.
    If the attachment has as "type" "file", then it is expected to have a field "file_path", with the path of file in the local file system. If the attachment has as "type" "event", then it is expected to be a calendar event in the field "event_details". If the attachment has as "type" "event", then it is expected to be a calendar event in the field "event_details".
    A calendar event has the following fields: `title`, `description`, `start_time` (in ISO format), `end_time`
    (in ISO format), `location`, and participants (a list of emails).
    :param recipient:  The email address of the recipient.
    :param cc: The list of email addresses to include in the CC field. If `null`, no email addresses are included.
    :param bcc: The list of email addresses to include in the BCC field. If `null`, no email addresses are included.
    """
    if username != username_groundtruth or password != password_groundtruth:
        raise ValueError("Login failed. Invalid username or password.")
    
    if attachments is not None:
        parsed_attachments = []
        for attachment in attachments:
            if not isinstance(attachment, dict):
                raise ValueError("Attachments must be dictionaries.")
            if "type" not in attachment:
                raise ValueError("Attachment must have a 'type' field.")
            if attachment["type"] == "file" or "file_path" in attachment or "file_id" in attachment:
                if ("file_path" not in attachment) and ("file_id" not in attachment):
                    raise ValueError("Attachment of type 'file' must have a 'file_path' or 'file_id' field.")
                if "file_path" in attachment:
                    file_id = upload_file_to_web(web_database=web_database, filesystem=filesystem, local_file_path=attachment["file_path"])
                    parsed_attachments.append(file_id)
                elif "file_id" in attachment:
                    file_id = attachment["file_id"]
                    parsed_attachments.append(file_id)
            else:
                if "event_details" not in attachment:
                    raise ValueError("Attachment of type 'event' must have an 'event_details' field.")
                parsed_attachments.append(CalendarEvent(**attachment["event_details"]))
    else:
        parsed_attachments = None

    sender_inbox = get_email_account_instance(email_database, sender)
    recipient_inboxs = []
    msg = ""

    if isinstance(recipients, str):
        recipients = [recipients]

    for recipient in recipients:
        if recipient is not None:
            cur_inbox = get_email_account_instance(email_database, recipient)
            if cur_inbox is not None:
                recipient_inboxs.append(cur_inbox)
            else:
                recipient_inboxs.append(email_database.add_new_address(recipient))
    
    if (sender_inbox is None) and (len(recipient_inboxs) == 0):
        raise ValueError("Email account invalid.")

    if sender_inbox is not None:
        msg = sender_inbox.send_email(recipients=recipients, subject=subject, body=body, attachments=parsed_attachments, cc=cc, bcc=bcc)
    
    for recipient_inbox in recipient_inboxs:
        msg = recipient_inbox.receive_email(sender, subject, body, parsed_attachments, cc, bcc)

    if return_str is not None:
        return return_str
    return msg

def send_email(
    email_database: Annotated[EmailDatabase, Depends("email_database")],
    filesystem: Annotated[Filesystem, Depends("filesystem")],
    web_database: Annotated[WebDatabase, Depends("web_database")],
    recipients: list[str],
    subject: str,
    body: str,
    attachments: list[dict] | None = None,
    cc: list[str] | None = None,
    bcc: list[str] | None = None,
) -> str:
    """Sends an email with the given `body` to the given `address`. Returns a dictionary with the email details.

    :param recipients: The list with the email addresses of the recipients.
    :param subject: The subject of the email.
    :param body: The body of the email.
    :param attachments: The list of attachments to include in the email. If `null`, no attachments are included.
    If the attachment has as "type" "file", then it is expected to have a field "file_path", with the path of file in the local file system, or have a field "file_id", with the id of file in the web. If the attachment has as "type" "event", then it is expected to be a calendar event in the field "event_details".
    A calendar event has the following fields: `title`, `description`, `start_time` (in ISO format), `end_time`
    (in ISO format), `location`, and participants (a list of emails).
    :param cc: The list of email addresses to include in the CC field. If `null`, no email addresses are included.
    :param bcc: The list of email addresses to include in the BCC field. If `null`, no email addresses are included.
    """
    if email_database.current_account is None:
        raise ValueError("No email account is login.")
    
    send_receive_email(email_database=email_database, filesystem=filesystem, web_database=web_database, sender=email_database.current_account, subject=subject, body=body, recipients=recipients, attachments=attachments, cc=cc, bcc=bcc)

    return "The email has been sent successfully."

def get_unread_emails(email_database: Annotated[EmailDatabase, Depends("email_database")]) -> list[Email]:
    """Returns all the unread emails in the inbox. Each email has a sender, a subject, and a body.
    The emails are marked as read after this function is called."""

    if email_database.current_account is None:
        raise ValueError("No email account is login.")
    inbox = get_email_account_instance(email_database, email_database.current_account)

    if inbox is None:
        return []
    return inbox.get_unread()


def search_emails(
    email_database: Annotated[EmailDatabase, Depends("email_database")],
    query: str,
    sender: str | None = None,
) -> list[Email]:
    """Searches for emails in the inbox that contain the given query in the subject or body. If `address` is provided,
    only emails from that address are searched.

    :param query: The query to search for in the email subject or body. If empty, all emails are returned.
    :param sender: The email address of the sender. If `null`, all emails are searched. Use `search_contacts_by_name`
    if the email address is unknown.
    """
    if email_database.current_account is None:
        raise ValueError("No email account is login.")
    inbox = get_email_account_instance(email_database, email_database.current_account)
    if inbox is None:
        return []
    matches = inbox.search_emails(query, sender)
    if len(matches) == 0:
        raise ValueError("No emails found. Try with a different query.")
    return matches


def delete_email(email_database: Annotated[EmailDatabase, Depends("email_database")], email_id: str) -> str:
    """Deletes the email with the given `email_id` from the inbox.

    :param email_id: The id of the email to delete.
    """
    if email_database.current_account is None:
        raise ValueError("No email account is login.")
    inbox = get_email_account_instance(email_database, email_database.current_account)
    if inbox is None:
        return "Email account not found."
    inbox.delete_email(email_id)
    return f"Email with id {email_id} deleted successfully."


def get_sent_emails(email_database: Annotated[EmailDatabase, Depends("email_database")]) -> list[Email]:
    """Returns all the sent emails in the inbox. Each email has a recipient, a subject, and a body."""
    if email_database.current_account is None:
        raise ValueError("No email account is login.")
    inbox = get_email_account_instance(email_database, email_database.current_account)
    if inbox is None:
        return []
    return inbox.sent


def get_received_emails(email_database: Annotated[EmailDatabase, Depends("email_database")]) -> list[Email]:
    """Returns all the received emails in the inbox. Each email has a sender, a subject, and a body."""
    if email_database.current_account is None:
        raise ValueError("No email account is login.")
    inbox = get_email_account_instance(email_database, email_database.current_account)
    if inbox is None:
        return []
    return inbox.received


def get_draft_emails(email_database: Annotated[EmailDatabase, Depends("email_database")]) -> list[Email]:
    """Returns all the draft emails in the inbox. Each email has a recipient, a subject, and a body."""
    if email_database.current_account is None:
        raise ValueError("No email account is login.")
    inbox = get_email_account_instance(email_database, email_database.current_account)
    if inbox is None:
        return []
    return inbox.drafts


def search_contacts_by_name(email_database: Annotated[EmailDatabase, Depends("email_database")], query: str) -> list[EmailContact]:
    """Finds contacts in the inbox's contact list by name.
    It returns a list of contacts that match the given name.

    :param query: The name of the contacts to search for.
    """
    if email_database.current_account is None:
        raise ValueError("No email account is login.")
    inbox = get_email_account_instance(email_database, email_database.current_account)
    if inbox is None:
        return []
    return inbox.find_contacts_by_name(query)


def search_contacts_by_email(email_database: Annotated[EmailDatabase, Depends("email_database")], query: str) -> list[EmailContact]:
    """Finds contacts in the inbox's contact list by email.
    It returns a list of contacts that match the given email.

    :param query: The email of the contacts to search for.
    """
    if email_database.current_account is None:
        raise ValueError("No email account is login.")
    inbox = get_email_account_instance(email_database, email_database.current_account)
    if inbox is None:
        return []
    return inbox.find_contacts_by_email(query)
