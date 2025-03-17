from prefect import flow, task
import random

@task
def check_email_folder():
    # Simulate checking an email folder in Microsoft 365
    # In a real scenario, use Microsoft Graph API to check the email folder
    print("Checking email folder...")
    # Simulate finding a new report
    return random.choice([True, False])


@task
def create_document_in_elastic():
    # Simulate creating a document in Elastic index with summary information
    print("Creating summary document in Elastic...")
    # Elastic code to index document goes here


@task
def queue_attachment_for_processing():
    # Simulate queueing attachment for processing
    print("Queueing attachment for processing...")
    # Add the job to a processing queue, possibly using a message queue system

@flow(log_prints=True, retries=3, retry_delay_seconds=5)
def email_check_and_queue():
    check_email_folder()
    create_document_in_elastic()
    queue_attachment_for_processing()

if __name__ == "__main__":
    email_check_and_queue()
