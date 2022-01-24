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

    client = get_docker_client()
    print(f'Client is {client}')


    container = create_container(client)
    container.start()

    tar_stream = create_tar_stream([
        ('cona.csv', 'bananana'.encode())
        ])

    #container.exec_run('touch cona.csv', user='mluser', workdir='/home/mluser')

    container.put_archive('/home/mluser/', tar_stream)
    print(container.status)
    print(container.exec_run('cat cona.csv', user='mluser', workdir='/home/mluser'))
    #print(container.status)
    container.kill()
    container.stop()
    container.remove()





if __name__ == '__main__':
    main()

