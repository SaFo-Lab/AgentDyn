import datetime

from pydantic import BaseModel, EmailStr, Field, model_validator
from typing_extensions import Self

from agentdojo.strenum import StrEnum

from typing import Any, Dict, Literal, Union

class Permissions(BaseModel):
    read: bool = True
    write: bool = False
    execute: bool = False

class File(BaseModel):
    type: Literal["file"] = "file"
    content: str
    permissions: Permissions = Field(default_factory=Permissions)

class Directory(BaseModel):
    type: Literal["directory"] = "directory"
    children: Dict[str, Union[File, "Directory"]] = Field(default_factory=dict)
    permissions: Permissions = Field(
        default_factory=lambda: Permissions(write=True)
    )


class WebFunction(BaseModel):
    handler: str
    static_parameters: dict[str, Any]


class Web(BaseModel):
    web_url: str
    web_content: str
    field_to_parameters: dict = Field(default_factory=dict)
    web_function: list[WebFunction] = Field(default_factory=list)
    web_requests: list[str] = Field(default_factory=list)
    download_source: str | None = None


WebFileID = str

class WebFile(File):
    source_name: str


class WebDirectory(Directory):
    source_name: str




class SharingPermission(StrEnum):
    r = "r"
    rw = "rw"


CloudDriveFileID = str


class CloudDriveFile(BaseModel):
    id_: CloudDriveFileID = Field(description="The unique identifier of the file")
    filename: str = Field(description="The name of the file")
    content: str = Field(description="The content of the file")
    owner: EmailStr = Field(description="The email of the owner of the file")
    last_modified: datetime.datetime = Field(description="The last modified timestamp")
    shared_with: dict[EmailStr, SharingPermission] = Field(
        default_factory=dict,
        description="The object containing emails with their sharing permissions",
    )
    size: int = Field(default=0, description="The size of the file in bytes")

    @model_validator(mode="after")
    def compute_size(self) -> Self:
        self.size = len(self.content)
        return self


class EvenStatus(StrEnum):
    confirmed = "confirmed"
    canceled = "canceled"


CalendarEventID = str


class CalendarEvent(BaseModel):
    id_: CalendarEventID = Field(description="The unique identifier of the event")
    title: str = Field(description="The title of the event")
    description: str = Field(description="The description of the event")
    start_time: datetime.datetime = Field(description="The start time of the event")
    end_time: datetime.datetime = Field(description="The end time of the event")
    location: str | None = Field(description="The location of the event")
    participants: list[EmailStr] = Field(description="The list of the emails of the participants")
    all_day: bool = Field(default=False, description="Whether the event is all day")
    status: EvenStatus = Field(description="The status of the event", default=EvenStatus.confirmed)


class EmailStatus(StrEnum):
    sent = "sent"
    received = "received"
    draft = "draft"


class EmailContact(BaseModel):
    email: EmailStr = Field(description="The email of the contact")
    name: str = Field(description="The name of the contact")


EmailID = str


class Email(BaseModel):
    id_: EmailID = Field(description="The unique identifier of the email")
    sender: EmailStr = Field(description="The email of the sender")
    recipients: list[EmailStr] = Field(description="The list of the emails of the recipients")
    cc: list[EmailStr] = Field(default_factory=list, description="The list of the emails of the CC recipients")
    bcc: list[EmailStr] = Field(default_factory=list, description="The list of the emails of the BCC recipients")
    subject: str = Field(description="The subject of the email")
    body: str = Field(description="The body of the email")
    status: EmailStatus = Field(description="The status of the email", default=EmailStatus.sent)
    read: bool = Field(default=False, description="Whether the email has been read")
    timestamp: datetime.datetime = Field(
        default_factory=datetime.datetime.now, description="The timestamp of the email"
    )
    attachments: list[CloudDriveFileID | CalendarEvent] = Field(
        default_factory=list,
        description="The list of attachments of the email. If the attachment is a file, it will be the ID of the file. If the attachment is a calendar event, it will be a CalendarEvent object.",
    )

