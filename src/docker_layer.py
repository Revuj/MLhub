import docker
import docker.errors
import io
import tarfile
import time




def get_docker_client():

    try:
        return docker.from_env()
    except docker.errors.DockerException:
        return None



def create_container(client):
    cont = 'josesilva69420/mlhub:latest'
    return client.containers.create(cont, command='sh', tty=True)


def create_tar_stream(files):

    tar_stream = io.BytesIO()

    tar = tarfile.TarFile(fileobj=tar_stream, mode='w')

    for name, content in files:
        info = tarfile.TarInfo(name=name)
        info.size = len(content)
        info.mtime = time.time()

        tar.addfile(info, io.BytesIO(content))
    tar.close()
    tar_stream.seek(0)
    return tar_stream

def main():
    pass





if __name__ == '__main__':
    main()

