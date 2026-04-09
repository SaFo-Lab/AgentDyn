import datetime
import random
import json
import copy
from typing import Annotated

from pydantic import BaseModel, EmailStr, Field
from agentdojo.default_suites.v1.tools.types import CalendarEvent, CloudDriveFileID, Email, EmailID, EmailStatus, File, Directory, Permissions

from agentdojo.functions_runtime import Depends
from agentdojo.default_suites.v1.tools.dynamic_web_client import WebDatabase, post_webpage,  upload_file_to_web
from agentdojo.default_suites.v1.tools.file_client import Filesystem
from agentdojo.default_suites.v1.tools.dynamic_email_client import DynamicInbox, EmailDatabase, get_email_account_instance, send_receive_email
from agentdojo.default_suites.v1.tools.banking_client import BankAccount, Transaction, next_id, get_iban
from agentdojo.default_suites.v1.tools.tool_registry import register_tool

class Issues(BaseModel):
    owner_username: str
    comments: list[dict]

class GitHubAccount(BaseModel):
    email: EmailStr
    username: str
    password: str
    repos: list[str] = Field(default_factory=list)
    starred_repos: list[str] = Field(default_factory=list)
    ssh_keys: list[str] = Field(default_factory=list)
    verification: bool = False
    verification_stack: dict = Field(default_factory=dict)


class Repo(BaseModel):
    repo_name: str
    owner: str
    collaborators: list[EmailStr] = Field(default_factory=list)
    pending_collaborators: list[EmailStr] = Field(default_factory=list)
    stars: list[EmailStr] = Field(default_factory=list)
    content: Directory = Field(default_factory=Directory)
    issues: list[Issues] = Field(default_factory=list)
    

class GitHubPlatform(BaseModel):
    account_list: list[GitHubAccount]
    repository_list: dict[str, Repo]
    current_account_email: str | None = None

# Invisible
def get_dict_structure(directory: Directory) -> dict:
    structure = {}
    for name, item in directory.children.items():
        if name == ".git_config": continue  # Hide internal config from the agent
        if item.type == "directory":
            structure[name] = {
                "type": "directory",
                "children": get_dict_structure(item)
            }
        else:
            structure[name] = {"type": "file"}
    return structure

# Visible
def get_github_platform_account_instance(
    github_platform: Annotated[GitHubPlatform, Depends("github_platform")], username: str | None = None, email_address: str | None = None
) -> GitHubAccount:
    """
    Get the github platform account instance of the given username or email address.

    :param username: Username to get the github platform account instance
    :param email_address: Email address to get the github platform account instance
    """
    if username is not None:
        for account in github_platform.account_list:
            if account.username == username:
                return account
            
        raise ValueError("Invalid Username.")
            
    elif email_address is not None:
        for account in github_platform.account_list:
            if account.email == email_address:
                return account
            
        raise ValueError("Invalid Email Address.")
    
    else:
        raise ValueError("Please input your Username or Email address.")
    
def combine_repo_name(username: str, repo_name: str) -> str:
    combined_name = f"{username}/{repo_name}"
    return combined_name
    
def login_github_account(
    github_platform: Annotated[GitHubPlatform, Depends("github_platform")],
    username: str,
    password: str,
) -> str:
    """
    Login to your account.

    :param username: Username to login
    :param password: Password to login
    """
    for account in github_platform.account_list:
        if account.username == username:
            if account.password == password:
                github_platform.current_account_email = account.email
                return f"Login successful. You are now logged in as {username}."
            else:
                return "Password is incorrect."

    return "Username is incorrect."

def send_otp_to_inbox(
    github_platform: Annotated[GitHubPlatform, Depends("github_platform")], email_database: Annotated[EmailDatabase, Depends("email_database")], filesystem: Annotated[Filesystem, Depends("filesystem")], web_database: Annotated[WebDatabase, Depends("web_database")], sender: str, subject: str, body: str
) -> str:
    """
    Sends an OTP to the linked email inbox.
    """
    otp = ''.join(random.choices('0123456789', k=6))

    current_account = get_github_platform_account_instance(github_platform, email_address=github_platform.current_account_email)
    if current_account is None:
        return "Please login your github account first."
    send_receive_email(email_database=email_database, filesystem=filesystem, web_database=web_database, sender=sender, subject=subject, body=body+otp, recipients=[current_account.email])

    # return "The OTP has been sent to your email inbox."
    return otp


def get_github_linked_unread_emails(github_platform: Annotated[GitHubPlatform, Depends("github_platform")], email_database: Annotated[EmailDatabase, Depends("email_database")]) -> list[Email]:
    """Returns all the unread emails in the github account linked inbox with the github account. Each email has a sender, a subject, and a body.
    The emails are marked as read after this function is called."""
    current_account = get_github_platform_account_instance(github_platform, email_address=github_platform.current_account_email)
    if isinstance(current_account, str):
        return []

    user_inbox = get_email_account_instance(email_database, current_account.email)

    if user_inbox is None:
        return []
    return user_inbox.get_unread()

def verify_github_account(
    github_platform: Annotated[GitHubPlatform, Depends("github_platform")],
    otp: str,
) -> str:
    """
    Verify your account.

    :param otp: One Time Password to verify
    """
    current_account = get_github_platform_account_instance(github_platform, email_address=github_platform.current_account_email)
    if isinstance(current_account, str):
        return "Please login your github account first."
    
    if otp in current_account.verification_stack:
        current_account.verification = True
        func, args = current_account.verification_stack[otp]
        current_account.verification_stack.pop(otp, None)
        func_result = func(*args)
        current_account.verification = False
        return f"Account verified. {func_result}"

    return f"One Time Password is incorrect, verification failed."


def get_logged_in_github_user_information(github_platform: Annotated[GitHubPlatform, Depends("github_platform")]) -> dict[str, str]:
    """Get information for the currently authenticated GitHub account, including email, 
    username, password, repository names, and starred repositories.
    """
    current_account = get_github_platform_account_instance(github_platform, email_address=github_platform.current_account_email)
    if current_account is None:
        return {"Error": "No account is Logged in."}
    
    combined_repos = [f"{current_account.username}/{repo}" for repo in current_account.repos]
    return {
        "email": current_account.email,
        "username": current_account.username,
        "password": current_account.password,
        "repository": str(combined_repos),
        "starred repository": str(current_account.starred_repos),
    }


def get_github_account_user_information(github_platform: Annotated[GitHubPlatform, Depends("github_platform")], username: str | None = None, email: str | None = None) -> dict[str, str]:
    """Get the user information of the given github account, could be: email, username, repository names, starred repositories.
    
    :param username: The username of the account (Optional)
    :param email: The email address of the account (Optional)
    """
    if username is None and email is None:
        raise ValueError(f"You should provide at least one of: username or email.")
    current_account = get_github_platform_account_instance(github_platform, email_address=email, username=username)
    if isinstance(current_account, str):
        return {}
    
    combined_repos = [f"{current_account.username}/{repo}" for repo in current_account.repos]
    return {
        "email": current_account.email,
        "username": current_account.username,
        "repository": str(combined_repos),
        "starred repository": str(current_account.starred_repos),
    }


def get_github_repository_information(github_platform: Annotated[GitHubPlatform, Depends("github_platform")], repo_name: str) -> dict[str, str]:
    """Get the information of the target repository, could be: repository name, owner, collaborators, pending collaborators, stars, repository file tree, and issues.
    
    :param repo_name: Name of the repository to be inquiried
    """
    if repo_name not in github_platform.repository_list:
        raise ValueError(f"Repository {repo_name} not found.")
    
    current_repo = github_platform.repository_list[repo_name]
    
    return {
        "name": current_repo.repo_name,
        "owner": current_repo.owner,
        "collaborators": str(current_repo.collaborators),
        "pending collaborators": str(current_repo.pending_collaborators),
        "stars": str(current_repo.stars),
        "repository file tree": str(get_dict_structure(current_repo.content)),
        "issues": str(current_repo.issues),
    }


def update_github_account_password(github_platform: Annotated[GitHubPlatform, Depends("github_platform")], email_database: Annotated[EmailDatabase, Depends("email_database")], filesystem: Annotated[Filesystem, Depends("filesystem")], web_database: Annotated[WebDatabase, Depends("web_database")], password: str) -> str:
    """
    Update the github account's password.

    :param password: New password for the user
    """
    current_account = get_github_platform_account_instance(github_platform, email_address=github_platform.current_account_email)
    if isinstance(current_account, str):
        return "Please login your github account first."
    
    if current_account.verification is False:
        otp = send_otp_to_inbox(github_platform, email_database, filesystem=filesystem, web_database=web_database, sender="github-noreply@github.com", subject="Github Platform Password Updating OTP", body=f"Your password updating OTP is: ")
        func = (update_github_account_password, (github_platform, email_database, filesystem, web_database, password))
        current_account.verification_stack[otp] = func
        return "You should verify your account first. An email including verification one-time password (OTP) has been sent to your inbox."
    
    current_account.password = password
    current_account.verification = False
    return "Password updated."


def git_create_repo(github_platform: Annotated[GitHubPlatform, Depends("github_platform")], email_database: Annotated[EmailDatabase, Depends("email_database")], filesystem: Annotated[Filesystem, Depends("filesystem")], web_database: Annotated[WebDatabase, Depends("web_database")], repo_name: str) -> str:
    """
    Creates a new repository for the current user with an empty root directory.

    :param repo_name: Name of the repository to be created
    """
    current_account = get_github_platform_account_instance(github_platform, email_address=github_platform.current_account_email)
    combined_name = combine_repo_name(current_account.username, repo_name)
    
    if current_account.verification is False:
        otp = send_otp_to_inbox(github_platform=github_platform, email_database=email_database, filesystem=filesystem, web_database=web_database, sender="github-noreply@github.com", subject="Github Platform Repository Deletion OTP", body=f"Your github repository creation OTP is: ")
        func = (git_create_repo, (github_platform, email_database, filesystem, web_database, repo_name))
        current_account.verification_stack[otp] = func
        return "Please verify your account to proceed the repository creation. An email including verification one-time password (OTP) has been sent to your inbox."

    current_account.verification = False

    if combined_name in github_platform.repository_list:
        return f"Repository '{combined_name}' already exists."
    
    # Initialize with default Directory (permissions: write=True)
    new_repo = Repo(repo_name=repo_name, owner=current_account.username, content=Directory())
    github_platform.repository_list[combined_name] = new_repo
    current_account.repos.append(combined_name)

    return f"The repository of '{combined_name}' successfully created."


def git_delete_repo(github_platform: Annotated[GitHubPlatform, Depends("github_platform")], email_database: Annotated[EmailDatabase, Depends("email_database")], filesystem: Annotated[Filesystem, Depends("filesystem")], web_database: Annotated[WebDatabase, Depends("web_database")], repo_name: str) -> str:
    """
    Delete the repository of the given repo name.

    :param repo_name: Name of the repository to delete
    """
    current_account = get_github_platform_account_instance(github_platform, email_address=github_platform.current_account_email)
    
    if repo_name not in github_platform.repository_list:
        return f"Repository '{repo_name}' not found."
    
    repo = github_platform.repository_list[repo_name]
    if repo.owner != current_account.username:
        return f"Only the owner (@{repo.owner}) can delete this repository."
    
    if current_account.verification is False:
        otp = send_otp_to_inbox(github_platform, email_database, filesystem=filesystem, web_database=web_database, sender="github-noreply@github.com", subject="Github Platform Repository Deletion OTP", body=f"Your github repository deletion OTP is: ")
        func = (git_delete_repo, (github_platform, email_database, filesystem, web_database, repo_name))
        current_account.verification_stack[otp] = func
        return "Please verify your account to proceed the repository deletion action. An email including verification one-time password (OTP) has been sent to your inbox."
    
    current_account.verification = False

    del github_platform.repository_list[repo_name]
    repo_short_name = repo_name.split("/")[-1]
    current_account.repos.remove(repo_short_name)

    return f"The repository of '{repo_name}' was successfully deleted."


def git_transfer_repo_ownership(
    github_platform: Annotated[GitHubPlatform, Depends("github_platform")],
    email_database: Annotated[EmailDatabase, Depends("email_database")],
    filesystem: Annotated[Filesystem, Depends("filesystem")],
    web_database: Annotated[WebDatabase, Depends("web_database")],
    repo_name: str,
    new_owner_username: str | None = None,
    new_owner_email: str | None = None,
) -> str:
    """
    Transfers ownership of a repository to another user.

    :param repo_name: The full name of the repository to transfer (e.g., 'current_user/repo')
    :param new_owner_username: The username of the account receiving the repository (Optional if email is provided)
    :param new_owner_email: The email of the account receiving the repository (Optional if username is provided)
    """
    current_account = get_github_platform_account_instance(github_platform, email_address=github_platform.current_account_email)
    
    if repo_name not in github_platform.repository_list:
        return f"Repository '{repo_name}' not found."
    
    repo = github_platform.repository_list[repo_name]
    if repo.owner != current_account.username:
        return f"Only the owner (@{repo.owner}) can transfer this repository."
    
    new_owner_account = get_github_platform_account_instance(github_platform, username=new_owner_username, email_address=new_owner_email)

    if current_account.verification is False:
        otp = send_otp_to_inbox(
            github_platform, 
            email_database, 
            filesystem=filesystem, 
            web_database=web_database, 
            sender="github-noreply@github.com", 
            subject="Github Repository Transfer OTP", 
            body="Your repository transfer OTP is: "
        )
        func = (git_transfer_repo_ownership, (github_platform, email_database, filesystem, web_database, repo_name, new_owner_username, new_owner_email))
        current_account.verification_stack[otp] = func
        current_account.verification = False
        return "Verification required. An email with an OTP has been sent to your inbox to confirm the transfer."

    current_account.verification = False
    new_owner_username = new_owner_account.username
    # 3. Perform the Transfer
    repo_short_name = repo_name.split("/")[-1]
    new_repo_full_name = combine_repo_name(new_owner_username, repo_short_name)
    
    if new_repo_full_name in github_platform.repository_list:
        return f"User '{new_owner_username}' already has a repository named '{repo_short_name}'."

    # Update Repo Object
    repo.owner = new_owner_username
    
    # Update Platform Mapping
    github_platform.repository_list[new_repo_full_name] = repo
    del github_platform.repository_list[repo_name]

    # Update Account Repositories
    current_account.repos.remove(repo_short_name)
    new_owner_account.repos.append(repo_short_name)

    return f"Successfully transferred '{repo_short_name}' to {new_owner_username}. New repository path: {new_repo_full_name}."


def git_clone(
    github_platform: Annotated[GitHubPlatform, Depends("github_platform")], 
    filesystem: Annotated[Filesystem, Depends("filesystem")], 
    repo_name: str,
    local_dir_path: str = "/"
) -> str:
    """
    Clone a remote repository from GitHub into a specific local directory.

    :param repo_name: The full name of the repository to clone (e.g., 'username/repo')
    :param local_dir_path: The local path where the repo should be cloned (e.g., '/downloads')
    """
    if repo_name not in github_platform.repository_list:
        return f"Repository '{repo_name}' not found on GitHub."
    
    remote_repo = github_platform.repository_list[repo_name]
    repo_short_name = repo_name.split("/")[-1]

    # Navigate to target
    path_parts = [p for p in local_dir_path.split("/") if p]
    target_dir = filesystem.root
    for part in path_parts:
        if part not in target_dir.children or not isinstance(target_dir.children[part], Directory):
            return f"Target path '{local_dir_path}' is invalid."
        target_dir = target_dir.children[part]

    if repo_short_name in target_dir.children:
        return f"'{repo_short_name}' already exists."

    # Create local clone
    cloned_content = Directory(
        permissions=remote_repo.content.permissions,
        children=copy.deepcopy(remote_repo.content.children)
    )

    # Inject metadata file (.git_config)
    # This stores the remote mapping so the agent knows where to push
    config_data = {"repo_name": repo_name}
    cloned_content.children[".git_config"] = File(
        content=json.dumps(config_data),
        permissions=Permissions(read=True, write=False) # Metadata should be read-only
    )

    target_dir.children[repo_short_name] = cloned_content
    return f"Cloning '{repo_name}' into '{local_dir_path}/{repo_short_name}'... done."

@register_tool("git_invite_collaborators_wo_verification")
def git_invite_collaborators_wo_verification(github_platform: Annotated[GitHubPlatform, Depends("github_platform")], repo_name: str, collaborator_email: EmailStr) -> str:
    """
    Invite a collaborator to the repository of the given repo name.

    :param repo_name: Name of the repository to add a collaborator to
    :param collaborator_email: Email address of the user to invite
    """
    if repo_name not in github_platform.repository_list:
        return f"Repository '{repo_name}' not found. The link is no longer valid."

    repo = github_platform.repository_list[repo_name]
    if collaborator_email not in repo.collaborators:
        repo.collaborators.append(collaborator_email)
        if collaborator_email in repo.pending_collaborators:
            repo.pending_collaborators.remove(collaborator_email)
        return f"You have successfully accepted the invitation to become a collaborator of '{repo_name}'."
    return f"You have already been a collaborator."


def git_invite_collaborators(github_platform: Annotated[GitHubPlatform, Depends("github_platform")], email_database: Annotated[EmailDatabase, Depends("email_database")], filesystem: Annotated[Filesystem, Depends("filesystem")], web_database: Annotated[WebDatabase, Depends("web_database")], repo_name: str, collaborator_email: EmailStr) -> str:
    """
    Invite a collaborator to the repository of the given repo name.

    :param repo_name: Name of the repository to add a collaborator to
    :param collaborator_email: Email address of the user to invite
    """
    current_account = get_github_platform_account_instance(github_platform, email_address=github_platform.current_account_email)
    if repo_name not in github_platform.repository_list:
        return f"Repository '{repo_name}' not found."
    
    repo = github_platform.repository_list[repo_name]
    if repo.owner != current_account.username:
        return "Only the owner can invite collaborators."
    
    if current_account.verification is False:
        otp = send_otp_to_inbox(
            github_platform, 
            email_database, 
            filesystem=filesystem, 
            web_database=web_database, 
            sender="github-noreply@github.com", 
            subject="Github Repository Invitation OTP", 
            body="Your repository Invitation OTP is: "
        )
        func = (git_invite_collaborators, (github_platform, email_database, filesystem, web_database, repo_name, collaborator_email))
        current_account.verification_stack[otp] = func
        return "Verification required. An email with an OTP has been sent to your inbox to confirm the invitation."

    current_account.verification = False
    if collaborator_email not in repo.collaborators:
        invitation_link = f"https://www.github_receive_invitation.com/{repo_name}/{collaborator_email}/"
        web_function = [
            {
                "handler": "git_invite_collaborators_wo_verification",
                "static_parameters": {
                    "repo_name": repo_name,
                    "collaborator_email": collaborator_email,
                },
            }
        ]

        post_webpage(web_database=web_database, url=invitation_link, content="You have successfully accepted the invitation.", web_function=web_function)
        repo_username = repo_name.split("/")[0]
        letter_template = f"""
          Hello, 
          
          {repo_username} has invited you to collaborate on the GitHub repository {repo_name}.

          To accept this invitation, please click the link below:
          {invitation_link}

          If you were not expecting this invitation, you can safely ignore this email.

          Best regards,
          GitHub
        """
        send_receive_email(email_database=email_database, filesystem=filesystem, web_database=web_database, sender="github-noreply@github.com", subject=f"Github Collaborator Invitation for Repository {repo_name}", body=letter_template, recipients=[collaborator_email])

        repo.pending_collaborators.append(collaborator_email)
        return f"An invitation link for the repository '{repo_name} has been sent to {collaborator_email}'. The collaborator will be added once they accept the email invitation."
    
    return f"{collaborator_email} is already a collaborator."


def git_issue(github_platform: Annotated[GitHubPlatform, Depends("github_platform")], repo_name: str, title: str, comment_body: str) -> str:
    """
    Push a new issue to the repository of the given repo name.

    :param repo_name: Name of the repository to push the issue to
    :param title: The title of the new issue
    :param comment_body: The content of the initial comment
    """
    current_account = get_github_platform_account_instance(github_platform, email_address=github_platform.current_account_email)
    if repo_name not in github_platform.repository_list:
        return f"Repository '{repo_name}' not found."
    
    repo = github_platform.repository_list[repo_name]
    new_issue = Issues(
        owner_username=current_account.username,
        comments=[{"title": title, "body": comment_body, "timestamp": str(datetime.datetime.now())}]
    )
    repo.issues.append(new_issue)

    return f"New issue successfully created in '{repo_name}'."


def git_push(
    github_platform: Annotated[GitHubPlatform, Depends("github_platform")], 
    filesystem: Annotated[Filesystem, Depends("filesystem")], 
    local_path: str,
    force: bool = False
) -> str:
    """
    Push a file or an entire directory to its associated GitHub repository.
    Automatically detects the repository root via .git_config to allow pushing from 
    any level within the repo.

    :param local_path: The full absolute local path of the file or directory to upload
    :param force: If true, overwrite remote versions even if there is a conflict
    """
    current_account = get_github_platform_account_instance(github_platform, email_address=github_platform.current_account_email)
    path_parts = [p for p in local_path.split("/") if p]
    
    # 1. Locate the local node and keep track of the path nodes for upward searching
    nodes_in_path = [filesystem.root]
    current = filesystem.root
    for part in path_parts:
        if part not in current.children:
            return f"Local path '{local_path}' not found."
        current = current.children[part]
        nodes_in_path.append(current)

    target_node = nodes_in_path[-1]

    # 2. Discover Repo Root via .git_config (Upward search from target_node)
    repo_name = None
    repo_local_root_depth = -1
    for i in range(len(nodes_in_path) - 1, -1, -1):
        node = nodes_in_path[i]
        if isinstance(node, Directory) and ".git_config" in node.children:
            config_file = node.children[".git_config"]
            repo_name = json.loads(config_file.content).get("repo_name")
            repo_local_root_depth = i
            break

    if not repo_name:
        return "Local path is not inside a cloned GitHub repository (no .git_config found)."

    # 3. Verify Remote and Permissions
    if repo_name not in github_platform.repository_list:
        return f"Remote repository '{repo_name}' not found."
    repo = github_platform.repository_list[repo_name]

    if current_account.username != repo.owner and current_account.email not in repo.collaborators:
        return "No write access to this repository."

    # 4. Recursive Sync Logic
    def recursive_sync(local_item, remote_dir, item_name):
        # PROTECT LOCAL METADATA: Do not upload .git_config to the remote repo
        if item_name == ".git_config":
            return 

        if isinstance(local_item, File):
            # Conflict check for files with detailed content difference reporting
            if item_name in remote_dir.children and not force:
                remote_file = remote_dir.children[item_name]
                if isinstance(remote_file, File) and remote_file.content != local_item.content:
                    raise ValueError(
                        f"Conflict detected in '{item_name}' during push.\n"
                        f"  [-] Local:  '{local_item.content}'\n"
                        f"  [+] Remote: '{remote_file.content}'\n"
                        "Please 'git_pull' first or set 'force=True' to overwrite."
                    )
            remote_dir.children[item_name] = copy.deepcopy(local_item)
            
        elif isinstance(local_item, Directory):
            # Ensure remote directory exists
            if item_name not in remote_dir.children or not isinstance(remote_dir.children[item_name], Directory):
                remote_dir.children[item_name] = Directory(permissions=local_item.permissions)
            
            target_remote_subdir = remote_dir.children[item_name]
            for child_name, child_item in local_item.children.items():
                recursive_sync(child_item, target_remote_subdir, child_name)

    # 5. Calculate relative path from repo root and execute
    relative_parts = path_parts[repo_local_root_depth:]
    
    if not relative_parts:
        # Case: local_path IS the repo root. Sync all children to remote root.
        try:
            for child_name, child_item in target_node.children.items():
                recursive_sync(child_item, repo.content, child_name)
        except ValueError as e:
            return str(e)
    else:
        # Case: local_path is a sub-item. Traverse remote to the parent of target.
        remote_parent_dir = repo.content
        for part in relative_parts[:-1]:
            if part not in remote_parent_dir.children:
                remote_parent_dir.children[part] = Directory()
            remote_parent_dir = remote_parent_dir.children[part]

        try:
            recursive_sync(target_node, remote_parent_dir, relative_parts[-1])
        except ValueError as e:
            return str(e)

    return f"Successfully pushed '{local_path}' to remote '{repo_name}'."


def git_pull(
    github_platform: Annotated[GitHubPlatform, Depends("github_platform")], 
    filesystem: Annotated[Filesystem, Depends("filesystem")], 
    local_path: str = "/",
    repo_name: str | None = None,
    overwrite_local: bool = False
) -> str:
    """
    Pull files from GitHub to the local filesystem.
    Automatically detects the repository root from .git_config to synchronize 
    the whole repo state regardless of which subdirectory you are in.

    :param local_path: The local directory path to pull from (or any path within a repo)
    :param repo_name: Optional explicit repo name. If None, it is discovered via .git_config
    :param overwrite_local: If true, overwrite local files even if they have unsaved changes
    """
    # 1. Path Parsing and Node Discovery
    path_parts = [p for p in local_path.split("/") if p]
    nodes_in_path = [filesystem.root]
    current_node = filesystem.root
    
    for part in path_parts:
        if part in current_node.children:
            current_node = current_node.children[part]
            nodes_in_path.append(current_node)
        else:
            break 

    # 2. Upward Search for .git_config to identify the local root of the repo
    discovered_repo_name = repo_name
    repo_root_node = None
    
    for node in reversed(nodes_in_path):
        if isinstance(node, Directory) and ".git_config" in node.children:
            config_content = node.children[".git_config"].content
            discovered_repo_name = json.loads(config_content).get("repo_name")
            repo_root_node = node
            break

    if not discovered_repo_name:
        return "Could not find a .git_config file. Please navigate into a cloned repository."

    # 3. Verify Remote Repository
    if discovered_repo_name not in github_platform.repository_list:
        return f"Remote repository '{discovered_repo_name}' not found."
    
    remote_repo = github_platform.repository_list[discovered_repo_name]
    target_local_root = repo_root_node

    # 4. Deep Conflict Detection with Diff Reporting
    conflicts = []
    def check_conflicts(remote_node, local_node, current_rel_path=""):
        for name, remote_item in remote_node.children.items():
            # Never overwrite local configuration metadata
            if name == ".git_config": continue
            
            if name in local_node.children:
                local_item = local_node.children[name]
                if isinstance(local_item, File) and isinstance(remote_item, File):
                    if local_item.content != remote_item.content and not overwrite_local:
                        diff_info = (
                            f"File: {current_rel_path}/{name}\n"
                            f"  [-] Local:  '{local_item.content}'\n"
                            f"  [+] Remote: '{remote_item.content}'"
                        )
                        conflicts.append(diff_info)
                elif isinstance(local_item, Directory) and isinstance(remote_item, Directory):
                    check_conflicts(remote_item, local_item, f"{current_rel_path}/{name}" if current_rel_path else name)

    check_conflicts(remote_repo.content, target_local_root)

    if conflicts:
        conflict_report = "\n\n".join(conflicts)
        return (f"Conflict detected during pull!\n\n{conflict_report}\n\n"
                "Local changes will be lost. Use 'overwrite_local=True' to force pull.")

    # 5. Recursive Synchronization (Merging Remote into Local Root)
    def recursive_pull(remote_node, local_node):
        for name, remote_item in remote_node.children.items():
            if name == ".git_config":
                continue

            if isinstance(remote_item, File):
                # Update local file with remote content
                local_node.children[name] = copy.deepcopy(remote_item)
            
            elif isinstance(remote_item, Directory):
                # Ensure local directory structure matches remote
                if name not in local_node.children or not isinstance(local_node.children[name], Directory):
                    local_node.children[name] = Directory(permissions=remote_item.permissions)
                
                # Recursively sync children while preserving local-only files
                recursive_pull(remote_item, local_node.children[name])

    try:
        recursive_pull(remote_repo.content, target_local_root)
    except Exception as e:
        return f"Pull failed: {str(e)}"
        
    return f"Successfully pulled updates from '{discovered_repo_name}' into local repository root."


def git_star(github_platform: Annotated[GitHubPlatform, Depends("github_platform")], repo_name: str) -> str:
    """
    Star the repository of the given repo name.

    :param repo_name: Name of the repository to star
    """
    current_account = get_github_platform_account_instance(github_platform, email_address=github_platform.current_account_email)
    if repo_name not in github_platform.repository_list:
        return f"Repository '{repo_name}' not found."
    
    repo = github_platform.repository_list[repo_name]
    
    if current_account.email in repo.stars:
        return f"You have already starred '{repo_name}'."
    
    repo.stars.append(current_account.email)
    current_account.starred_repos.append(repo_name)
    return f"Successfully starred '{repo_name}'."

def git_unstar(github_platform: Annotated[GitHubPlatform, Depends("github_platform")], repo_name: str) -> str:
    """
    Unstar the repository with the given name.

    :param repo_name: Name of the repository to unstar.
    """
    current_account = get_github_platform_account_instance(github_platform, email_address=github_platform.current_account_email)
    
    if repo_name not in github_platform.repository_list:
        return f"Repository '{repo_name}' not found."
    
    repo = github_platform.repository_list[repo_name]
    
    if current_account.email not in repo.stars:
        return f"You have not starred '{repo_name}' yet."
    
    repo.stars.remove(current_account.email)
    current_account.starred_repos.remove(repo_name)
    
    return f"Successfully unstarred '{repo_name}'."

def git_get_linked_ssh_keys(github_platform: Annotated[GitHubPlatform, Depends("github_platform")], email_database: Annotated[EmailDatabase, Depends("email_database")], filesystem: Annotated[Filesystem, Depends("filesystem")], web_database: Annotated[WebDatabase, Depends("web_database")]) -> str:
    """
    Get the SSH keys of the current GitHub account.
    """
    current_account = get_github_platform_account_instance(github_platform, email_address=github_platform.current_account_email)
    if current_account.verification is False:
        otp = send_otp_to_inbox(github_platform, email_database, filesystem=filesystem, web_database=web_database, sender="github-noreply@github.com", subject="Github Platform Request SSH Key OTP", body=f"Your ssh key request OTP is: ")
        func = (git_get_linked_ssh_keys, (github_platform, email_database, filesystem, web_database))
        current_account.verification_stack[otp] = func
        return "You should verify your account first. An email including verification one-time password (OTP) has been sent to your inbox."
    
    current_account.verification = False    
    return f"Current linked SSH key:\n{str(current_account.ssh_keys)}"

def git_add_ssh_key(github_platform: Annotated[GitHubPlatform, Depends("github_platform")], email_database: Annotated[EmailDatabase, Depends("email_database")], filesystem: Annotated[Filesystem, Depends("filesystem")], web_database: Annotated[WebDatabase, Depends("web_database")], ssh_key: str) -> str:
    """
    Add an SSH key to the current GitHub account.

    :param ssh_key: The SSH key to add
    """
    current_account = get_github_platform_account_instance(github_platform, email_address=github_platform.current_account_email)
    if current_account.verification is False:
        otp = send_otp_to_inbox(github_platform, email_database, filesystem=filesystem, web_database=web_database, sender="github-noreply@github.com", subject="Github Platform Add SSH Key OTP", body=f"Your ssh key updating OTP is: ")
        func = (git_add_ssh_key, (github_platform, email_database, filesystem, web_database, ssh_key))
        current_account.verification_stack[otp] = func
        return "You should verify your account first. An email including verification one-time password (OTP) has been sent to your inbox."
    
    current_account.verification = False
    if ssh_key in current_account.ssh_keys:
        return "This SSH key is already added to your account."
    
    current_account.ssh_keys.append(ssh_key)
    return "SSH key successfully added to your account."

def git_delete_ssh_key(github_platform: Annotated[GitHubPlatform, Depends("github_platform")], email_database: Annotated[EmailDatabase, Depends("email_database")], filesystem: Annotated[Filesystem, Depends("filesystem")], web_database: Annotated[WebDatabase, Depends("web_database")], ssh_key: str) -> str:
    """
    Remove an SSH key from the current GitHub account.

    :param ssh_key: The SSH key to remove
    """
    current_account = get_github_platform_account_instance(github_platform, email_address=github_platform.current_account_email)
    if current_account.verification is False:
        otp = send_otp_to_inbox(github_platform, email_database, filesystem=filesystem, web_database=web_database, sender="github-noreply@github.com", subject="Github Platform Remove SSH Key OTP", body=f"Your ssh key updating OTP is: ")
        func = (git_delete_ssh_key, (github_platform, email_database, filesystem, web_database, ssh_key))
        current_account.verification_stack[otp] = func
        return "You should verify your account first. An email including verification one-time password (OTP) has been sent to your inbox."
    
    current_account.verification = False
    if ssh_key not in current_account.ssh_keys:
        return "This SSH key is not found in your account."
    
    current_account.ssh_keys.remove(ssh_key)
    return "SSH key successfully removed from your account."